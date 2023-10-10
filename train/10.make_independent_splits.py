import os
import requests
import glob


# Download
# https://cdn.rcsb.org/resources/sequence/clusters/clusters-by-entity-30.txt
# using the requests library

if not os.path.exists("clusters-by-entity-30.txt"):
    txt = requests.get(
        "https://cdn.rcsb.org/resources/sequence/clusters/clusters-by-entity-30.txt"
    ).text
    with open("clusters-by-entity-30.txt", "w") as f:
        f.write(txt)

# Load in all_cen.types data
with open("./all_cen.types", "r") as f:
    all_cen_data = {}
    for line in f:
        pdbid = line.split()[-1].split("/")[-2].upper()
        all_cen_data[pdbid] = line

# Get the pdbs to consider
pdbs_in_pdbbind = {k for k in all_cen_data.keys()}

# Go through the file and make a list of all the PDB codes

pdb_clusters = []
with open("clusters-by-entity-30.txt", "r") as f:
    for line in f:
        if line.startswith("#"):
            continue
        pdb_ids = line.strip().split()
        pdb_ids = [x.split("_")[0] for x in pdb_ids]
        pdb_ids = set(pdb_ids)
        pdb_clusters.append(pdb_ids)

# Count how many clusters each PDB id is in
pdb_counts = {}
for pdb_ids in pdb_clusters:
    for pdb_id in pdb_ids:
        if pdb_id not in pdb_counts:
            pdb_counts[pdb_id] = 0
        pdb_counts[pdb_id] += 1

# Make a list of all the PDB ids that are in more than one cluster
pdb_ids_in_more_than_one_cluster = []
for pdb_id, count in pdb_counts.items():
    if count > 1:
        pdb_ids_in_more_than_one_cluster.append(pdb_id)
pdb_ids_in_more_than_one_cluster = set(pdb_ids_in_more_than_one_cluster)

# Remove these pdb_ids from the list of clusters
new_pdb_clusters = []
for pdb_ids in pdb_clusters:
    new_pdb_ids = []
    for pdb_id in pdb_ids:
        if pdb_id not in pdb_ids_in_more_than_one_cluster:
            new_pdb_ids.append(pdb_id)
    new_pdb_ids = set(new_pdb_ids)
    if len(new_pdb_ids) > 0:
        new_pdb_clusters.append(new_pdb_ids)

# Remove any pdb that is not in pdbs_in_pdbbind from the list of clusters too
new_pdb_clusters2 = []
for pdb_ids in new_pdb_clusters:
    new_pdb_ids = []
    for pdb_id in pdb_ids:
        if pdb_id in pdbs_in_pdbbind:
            new_pdb_ids.append(pdb_id)
    new_pdb_ids = set(new_pdb_ids)
    if len(new_pdb_ids) > 0:
        new_pdb_clusters2.append(new_pdb_ids)

# Divide the pdbs into three groups
group1 = []
group2 = []
group3 = []
for pdb_cluster in new_pdb_clusters2:
    if len(group1) <= len(group2) and len(group1) <= len(group3):
        group1.extend(list(pdb_cluster))
    elif len(group2) <= len(group1) and len(group2) <= len(group3):
        group2.extend(list(pdb_cluster))
    elif len(group3) <= len(group1) and len(group3) <= len(group2):
        group3.extend(list(pdb_cluster))

# Print out the lengths of the clusters
print(len(group1))
print(len(group2))
print(len(group3))
print("Total:", len(group1) + len(group2) + len(group3))
print("Unique:", len(set(group1 + group2 + group3)))
print("Total in PDBBind used:", len(pdbs_in_pdbbind))
print("")

# Confirm no overlap
assert len(set(group1) & set(group2)) == 0
assert len(set(group1) & set(group3)) == 0
assert len(set(group2) & set(group3)) == 0

filenames_and_clusters = zip(
    ["independentsplit1", "independentsplit2", "independentsplit3"],
    [group1, group2, group3],
)

# Now create test and train sets. Each train set is two of the groups, with the
# corresponding test set being the third group.
train_test_sets = [
    ((group1, group2), group3),
    ((group1, group3), group2),
    ((group2, group3), group1),
]

for idx, (train, test) in enumerate(train_test_sets):

    # Merge the two train groups
    train = list(train[0]) + list(train[1])
    # Save the train set
    with open("independentsplittrain" + str(idx) + "_cen.types", "w") as f:
        for pdbid in train:
            f.write(all_cen_data[pdbid])

    # Save the test set
    with open("independentsplittest" + str(idx) + "_cen.types", "w") as f:
        for pdbid in test:
            f.write(all_cen_data[pdbid])

