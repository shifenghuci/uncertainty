# Goal
Trying to get a hang of minimal mech interp workflow.

This repo is heavily inspired by [Javier's repo](https://github.com/javiferran/sae_entities)

[Deepwiki link on Javier's repo](https://deepwiki.com/javiferran/sae_entities)

---

## Key libraries:
1. `transformerLens` - load hooked (monitored) transformer models.
2. `sae-lens` - load trained sae model

## Reference files from Javier's repo
1. `utils/sae-utils.py` - sae_encode and sae_decode func definition
2. `mech_interp/uncertain_features.py` - sae feature extraction
3. `mech_interp/hooks_.util.py` - steer_sae_latents, ablate_sae_latents

## Importance documentation
1. [ActivationCache, monitoring activation on each layer](https://transformerlensorg.github.io/TransformerLens/generated/code/transformer_lens.ActivationCache.html)


## Setup code
``` sh
uv init uncertainty
cd uncertainty
uv add transformer-lens sae-lens
uv lock
```

## Key lines
``` python
from transformer-lens import HookedTransformer
from sae-lens import SAE

#### Activation Monitoring ####

# load a specific model
model = HookedTransformer.from_pretrained(
  'meta-llama/Llama-3.1-8B-Instruct'
)

# run one forward pass (a.k.a inference)
logits, cache = model.run_with_cache(
  tokens, runtime_type = 'logits'
)

# access the input activations **into** the 15th attention blocks h
activations = cache['blocks.15.hook_resid_pre']

#### Feature Decomposition using SAE ####

# load sae for the specific layer of the hookedTransformers
sae, cfg_dict, sparsity = SAE.from_pretrain{
	release = 'llama_scope_lxr_8x', # this and sae-id is pretty random and organization dependent.
	sae-id = 'l15r_8x', # 15th layer, residual stream, 8x expansion factor
	device = 'mps' if torch.backends.mps.is_available() else 'cpu',
	# device = 'cuda' if torch.cuda_is_available() else 'cpu',
}

# decompose activations with SAE
sae_features = sae.encode(last_token_activation)

# filter out active feature (sae_feature should be sparse with most entry having the value of zero)
active_features = (sae_features > 0).nonzero(as_tuple=True)[1]

# We want to know the intensity of the activation on each non-zero feature
feature_activations = sae_features[0, active_feature]


#### Steering ####

```