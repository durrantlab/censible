Colab? Update link in README.md.

Mention output is pK(affinity) somewhere. Output in terms of mM?

# DONE

Need to add README.md. Done, but run through it from scratch to confirm works.

Need to update prepare_data/ files. Need to auto detect if user hasn't prepared
that directory. Need to test. Perhaps move everyhting to train, organize scripts
as best you can, but then say not supported.

To consider: at inference time, will users have to use gninatyper to prepare
their molecules? That seems impractical.

What about centering the grid on the ligand center of geometry?

https://github.com/gnina/gnina/blob/853704170de2b92d1208245e02419c680744b757/gninasrc/gninatyper/gninatyper.cpp

https://gnina.github.io/libmolgrid/python/index.html#the-gridmaker-class

