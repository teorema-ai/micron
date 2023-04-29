MICRON:
#...: datablocks: midatasets --> datasets, mimodels --> models and implement fs usage in datasets and models
#...:?datablocks: DB.list() should indicate version more clearly?  Currently only present in the shard_pathset string.

#...: datablocks: BUG: list fails for differet datablock versions, since scopes change schema.
#...: datablocks: BUG: done_time Response assignment in _done_callback is broken, or upstream/downstream extraction through Task/Report is.
#...: datablocks: BUG: Response.__tag__() seems broken



#...: ?datablocks: DB.define(): executes DB.define_request(**scope), which generates a Record with that scope.
#...:              NO need?: build() acts as a define+build if necessary and records()/list() should be sufficiet to inquire if an alias has been built.
#...: ?datablocks: Datablock.define --> Datablock.class

#...:!micron: incorporation of expression counts requires conditional learning
            goal: enable cancer subtype classification
            goal: conditional distribution that is more expressive than just the marginals (relative expression counts) of observed sequences
            goal: learn conditional representations that identify class (cancer subtype) from easier observables than all counts:
                e.g., top-k sequences by expression count
                      novel sequences
                      structure of top-k sequences (presence/absence of substrings)
#...:!micron: representation learning for realizable classifying feature set
            goal: identify most salient/predictive representation features
            goal: "invert" to obtain "efficient" measurable predictive concepts
            goal: "efficient" means 
                computable from small samples
                [easily extracted]
#...:!micron: model: downstream expression counts learning: similar to CornBERT
#...:!micron: model: learn expression counts --> cancer subtype classifier?  SVM?
#...:?micron: model: expression count: 
#...:        downstream head from sequence representation learning or another mode of representation learning?
#...:        compare to CornBERT

#...: micron: train GRCh38 model + finetune on MiRNA model
#...:
#...: datablocks: pool: create Ray from DATABLOCKS_RAY_HOST and DATABLOCKS_RAY_PORT by default.




* EMBED TRAINING set: 
    - Tokenize each sequence
    - Evaluate model on all of each of the sequence's initial segments, collecting the appropriate hidden states
    - Use training sequences long_latents and include them in UMAP      
* UMAP TRAIN & SAMPLE sets:
    - COMPARE distribution coverage
* UMAP
    - COLOR training set by cancer/patient prevalence 
        - pick the cancer where this sequence occurs the most
        - if there are not too many ties, break them arbitrarily
* PERPLEXITY/CROSS-ENTROPY measure of training set sequences
    - LITERATURE on how to evaluate the model performance by looking at perplexity
    - CODE for how to compute perplexity from model generation/transition scores
    - UNDERSTAND how CROSS-ENTROPY LOSS relates to PERPLEXITY
* PRETRAIN on other RNA datasets    
    
* REDUCE GPT2 parameter set

* DECONFOUNDING/separation/whitening in latent space
