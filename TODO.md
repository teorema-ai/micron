#...
* git commit
* Factor micron.dataset for Datablock.define():compute
* Run through datablocks.exec()
* Implement datablocks.run(), which should
	- build() the datablock
        	. record request.response.summary in a separate topic '.summary'
		. summary must include the hash of datablock's build() kwargs
		. summary must include the datablock instance's __init__() kwargs as 'config'
		. config must avoid nesting, instead summarizing datablock arguments by their classpath@hash
	- list() all summaries for a given datablock instance
	 

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
