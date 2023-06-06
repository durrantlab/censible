import molgrid
#from openbabel import pybel
import json
import argparse
import torch
import numpy as np
import subprocess

from CEN_model import CENet

# global termset dict -- defined by whoever trained the model (us)
term_sizes = {'all': 123,
                'smina': 21,
                'gaussian': 102}

def load_data_file(filename: str, dtype=np.bool_):
    # If it ends in npy, load it via numpy.
    if filename.endswith(".npy"):
        return np.load(filename)
    
    # # It must be a csv file.
    # with open(filename, "r") as f:
    #     txt = f.read()
    #     txt = txt.replace("\n", " ")
    #     while "  " in txt:
    #         txt = txt.replace("  ", " ")
    #     txt = txt.replace("[ ", "[").replace(" ]", "]")
    #     txt = txt.replace(" ", ",")
    #     txt = txt.replace("True", "true").replace("False", "false")
    #     txt = txt.replace("tensor(", "")
    #     txt = txt.split(",device=")[0]
    #     while ",," in txt:
    #         txt = txt.replace(",,", ",")
    #     txt = txt.replace("],", "]")
        
    #     data = json.loads(txt)
    # return np.array(data, dtype=dtype)

def load_example(lig_datapath, rec_datapath, terms_file, norm_factors_file):
    ### get the single_example_terms -- one set of smina computed terms 
    # load normalization term data
    terms_to_keep = load_data_file(terms_file, dtype=np.bool_)
    import pdb; pdb.set_trace()
    norm_factors = load_data_file(norm_factors_file, float)[terms_to_keep]

    # get CEN terms for proper termset
    # this is my smina path i neglected to append it
    smina_out = str(
        subprocess.check_output(
            f'../../smina --custom_scoring smina_ordered_terms.txt --score_only -r {rec_datapath} -l {lig_datapath}',
            shell=True,
        )
    )

    pf = lig_datapath.split('.')[-2].split('/')[-1]
    smina_computed_terms = smina_out[smina_out.find(pf):smina_out.find('\\nRefine time')][(len(pf)+1):]

    smina_outfile = 'types_file_cen.txt'
    with open(smina_outfile,'w') as smina_out_f:
        smina_out_f.write(
            f'{smina_computed_terms} '
            + rec_datapath.split('a/')[-1]
            + ' '
            + lig_datapath.split('a/')[-1]
        )
    smina_computed_terms = np.fromstring(smina_computed_terms,sep=' ')

    example = molgrid.ExampleProvider(data_root='../PDBbind16_data/full_data/',default_batch_size=1)
    example.populate(smina_outfile)

    return (example,terms_to_keep,norm_factors)

# load in model -- from torch
def load_model(modelpath,termset):
    dims = (28, 48, 48, 48)
    terms = term_sizes[termset]
    model = CENet(dims,terms)
    model.load_state_dict(torch.load(modelpath))
    model.eval()

    return model

# apply model to test data
def test_apply(example_data,terms_to_keep,norm_factors,model):
    gm = molgrid.GridMaker()
    norm_factors = torch.from_numpy(norm_factors).to('cuda')
    terms_to_keep = torch.from_numpy(terms_to_keep).to('cuda')

    single_label_for_testing = torch.zeros(
        (1, example_data.num_labels()), dtype=torch.float32, device="cuda"
    )
    single_input_for_testing = torch.zeros((1,) + (28, 48, 48, 48), dtype=torch.float32, device="cuda")
    
    test_batch = example_data.next_batch()
    # Get this batch's labels
    test_batch.extract_labels(single_label_for_testing)

    # Populate the single_input_for_testing tensor with an example.
    gm.forward(test_batch, single_input_for_testing)

    model.to('cuda')
    
    print('running model to predict')
    # Run that through the model.
    output, coef_predict, weighted_terms = model(
                    single_input_for_testing,
                    single_label_for_testing[:, :][:, terms_to_keep] * norm_factors
                )
    return (output, coef_predict, weighted_terms)

# run the model on test example
def get_cmd_args():
    # Create argparser 
    parser = argparse.ArgumentParser()
    parser.add_argument("--ligpath",required=True,help="path to the ligand")
    parser.add_argument("--recpath",required=True,help="path to the receptor")
    parser.add_argument("--modelpath",default='model.pt',help="path to the trained model (e.g., model.pt)")
    parser.add_argument("--term_set",default='all',help="the terms to use: all, smina, gaussian")
    parser.add_argument("--terms_file",default='terms_to_keep.npy',help="The terms-to-keep file (npy)")
    parser.add_argument("--norm_factors_file",default='all_normalization_factors.npy',help="the normalization factors file (npy)")
    parser.add_argument("--out_prefix",default='',help="prefix to use for saving")

    return parser.parse_args()

def test_run():
    l = '../PDBbind16_data/full_data/2r0h/2r0h_ligand.sdf'
    r = '../PDBbind16_data/full_data/2r0h/2r0h_pocket.pdb'
    norm = 'all_normalization_factors.npy'
    m = load_model('model.pt','all')
    inputs_se = load_example(l,r,'terms_to_keep.npy',norm)
    print('data and model loaded. applying')

    prediction, coef_predictions, weighted_ind_terms = test_apply(inputs_se[0],inputs_se[1],inputs_se[2],m)


### WHEN YOU RUN THIS ###
# Usage: python apply_model.py --ligpath <path to ligand file> --recpath <path to receptor file> --modelpath <path to model> 
#           --term_set <'all', 'smina', or 'gaussian'> --terms_file <npy file containing boolean array of terms to keep> 
#           --norm_factors_file <npy file containing array of normalization factors>
args = get_cmd_args()
# load the data
example, terms_to_keep, norm_factors = load_example(args.ligpath,args.recpath,args.terms_file,args.norm_factors_file)
# load the model
model = load_model(args.modelpath,args.term_set)
print('data and model loaded.')
prediction, coef_predictions, weighted_ind_terms = test_apply(example,terms_to_keep,norm_factors,model)

print(f"Affinity prediction: {str(round(float(prediction), 5))}")

if args.out_prefix == '':
    prefix = args.ligpath.split('/')[-1].split('.')[-2]+'_'+args.recpath.split('/')[-1].split('.')[-2]

np.savetxt(f'{prefix}_coeffs.txt', coef_predictions.cpu().detach().numpy())
np.savetxt(f'{prefix}_weighted.txt', weighted_ind_terms.cpu().detach().numpy())

print(f"Coefficients saved in: {prefix}_coeffs.txt")
print(f"Weighted terms saved in: {prefix}_weighted.txt")
