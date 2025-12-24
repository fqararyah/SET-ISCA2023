#include "nns/nns.h"

#include <cassert>


typedef TransposeLayer::dim Ldims;

static lid_t add_attention(
		Network& n, const std::string& name,
		len_t len, len_t numG, len_t gSize, len_t decode_len,
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
		len_t len, len_t numG, len_t gSize, len_t ff_len, len_t decode_len,
		lid_t prev
){
	lid_t next_prev;
	next_prev = add_attention(n, name, len, numG, gSize, decode_len, prev);
	prev = n.add(NLAYER(name + "_elt1", Eltwise, K=numG*gSize, H=len, W=1, N=2), {prev, next_prev});
	n.add(NLAYER(name + "_feedfwd1", Conv, C=numG*gSize, K=ff_len, H=len, W=1));
	next_prev = n.add(NLAYER(name + "_feedfwd2", Conv, C=ff_len, K=numG*gSize, H=len, W=1));
	return n.add(NLAYER(name + "_elt2", Eltwise, K=numG*gSize, H=len, W=1, N=2), {prev, next_prev});
}

static Network gen_llama(len_t num_layers, len_t seq_len, len_t numG=32, len_t gSize=128, len_t ff_len=11008){
	Network n;
	InputData input("input", fmap_shape(numG*gSize, seq_len, 1));

	lid_t prev = 0;
	n.add(NLAYER("embed", Conv, H=seq_len, W=1, C=numG*gSize), {}, 0, {input});

	for(len_t i=0; i<num_layers; ++i){
		prev = add_transformer_block(n, "layer_" + std::to_string(i), seq_len, numG, gSize, ff_len, 0, prev);
	}

	n.add(NLAYER("ln_f", Conv, C=numG*gSize, K=numG*gSize, H=seq_len, W=1), {prev});
	n.add(NLAYER("lm_head", FC, C=numG*gSize, K=32000)); // vocab size for LLaMA

	return n;
}

//const Network llama_7b = gen_llama(32, 2048); // 32 layers for LLaMA 7B

static Network create_transformer(
		len_t numG, len_t gSize, len_t nBlock, bool is_prefill,
		len_t vocab_len = 1000, len_t len = 512, len_t ff_len = 0
){
	// Default settings.
	if(ff_len == 0){
		ff_len = 4 * len;
	}

	len_t decode_len = 0;
	if(!is_prefill){
		decode_len = len;
		len = 1;
	}

	// Length of embedding
	len_t totG = numG * gSize;
	// Number of embedding
	len_t curH = len;

	lid_t block_prev;
	Network::layer_set prevs;
	Network n;

	InputData input_layer("input_layer", fmap_shape(totG, curH, 1));
	block_prev = n.add(NLAYER("word_embed", PTP, K=totG, H=curH, W=1), {}, 0, {input_layer});
	for(len_t i=1; i<=nBlock; ++i){
		block_prev = add_transformer_block(n, "block"+std::to_string(i), len, numG, gSize, ff_len, decode_len, block_prev);
	}
	n.add(NLAYER("proj", Conv, C=totG, K=vocab_len, H=curH, W=1), {block_prev});
	return n;
};

const Network llama_prefill_block = create_transformer(32, 128, 32, true, 32000, 2048, 11008);
const Network llama_decode_block = create_transformer(32, 128, 1, false, 32000, 2048, 11008);