#include "nns/nns.h"

#include <cassert>


typedef TransposeLayer::dim Ldims;

static lid_t add_attention(
		Network& n, const std::string& name,
		len_t len, len_t numG, len_t gSize,
		lid_t prev
){

	lid_t Q, K, V, QK, QK_elt, QKV;
	Network::layer_set Ks;
	Q = n.add(NLAYER(name + "_Q", Conv, H=len, W=1, C=numG*gSize), {prev});
	for(len_t i=0; i<numG; ++i){
		n.add(NLAYER(name + "_K" + std::to_string(i), Conv, C=numG*gSize, K=gSize, H=len, W=1), {prev});
		K = n.add(NLAYER(name + "_Kt" + std::to_string(i), Transpose, K=len, H=gSize, W=1, order[Ldims::C]=Ldims::H, order[Ldims::H]=Ldims::C));
		Ks.push_back(K);
	}
	K = n.add(NLAYER(name + "_K", PTP, K=numG*len, H=gSize, W=1), Ks);
	V = n.add(NLAYER(name + "_V", Conv, C=numG*gSize, H=len, W=1), {prev});
	QK = n.add(NLAYER(name + "_QK", GroupConv, H=len, W=1, C=numG*gSize, K = numG*len, G=numG), {Q}, 0, {}, {K});
	QK_elt = n.add(NLAYER(name + "_QK_elt", PTP, K=numG*len, H=len, W=1), {QK});
	QKV = n.add(NLAYER(name + "_QKV", GroupConv, H=len, W=1, C=numG*len, K=numG*gSize, G=numG), {QK_elt}, 0, {}, {V});
	return n.add(NLAYER(name + "_FC", Conv, H=len, W=1, C=numG*gSize), {QKV});
}

static lid_t add_transformer_block(
		Network& n, const std::string& name,
		len_t len, len_t numG, len_t gSize, len_t ff_len,
		lid_t prev
){
	lid_t next_prev;
	next_prev = add_attention(n, name, len, numG, gSize, prev);
	prev = n.add(NLAYER(name + "_elt1", Eltwise, K=numG*gSize, H=len, W=1, N=2), {prev, next_prev});
	n.add(NLAYER(name + "_feedfwd1", Conv, C=numG*gSize, K=ff_len, H=len, W=1));
	next_prev = n.add(NLAYER(name + "_feedfwd2", Conv, C=ff_len, K=numG*gSize, H=len, W=1));
	return n.add(NLAYER(name + "_elt2", Eltwise, K=numG*gSize, H=len, W=1, N=2), {prev, next_prev});
}

static Network gen_vit(len_t num_layers, len_t patch_size=16, len_t num_patches=196, len_t embed_dim=768, len_t num_heads=12, len_t ff_dim=3072){
	Network n;
	InputData input("input", fmap_shape(embed_dim, num_patches, 1));

	// Patch embedding + cls token
	lid_t prev = n.add(NLAYER("patch_embed", Conv, H=num_patches, W=1, C=embed_dim), {}, 0, {input});

	// Add position embedding (simplified, assuming learned)
	// In practice, add a constant or something, but here skip for simplicity

	for(len_t i=0; i<num_layers; ++i){
		prev = add_transformer_block(n, "layer_" + std::to_string(i), num_patches, num_heads, embed_dim/num_heads, ff_dim, prev);
	}

	n.add(NLAYER("ln_f", Conv, C=embed_dim, K=embed_dim, H=num_patches, W=1), {prev});
	lid_t pool = n.add(NLAYER("pool", Pooling, K=embed_dim, H=1, R=num_patches), {prev});
	n.add(NLAYER("head", FC, C=embed_dim, K=1000), {pool});
	return n;
}

const Network vit_base = gen_vit(12);
//const Network vit_large = gen_vit(24, 16, 196, 1024, 16, 4096);