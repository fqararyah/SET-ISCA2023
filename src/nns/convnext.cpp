#include "nns/nns.h"


static lid_t add_convnext_block(
	Network& n, lid_t prev,
	int stage, int blk,
	len_t C, len_t H_len
){
	std::string name = "conv_"+std::to_string(stage)+"_"+std::to_string(blk)+"_";
	lid_t layer1 = n.add(NLAYER(name+"dw", GroupConv, C=C, K=C, H=H_len, R=7, G=C), {prev});
	lid_t layer2 = n.add(NLAYER(name+"pw1", Conv, C=C, K=4*C, H=H_len), {layer1});
	lid_t layer3 = n.add(NLAYER(name+"pw2", Conv, C=4*C, K=C, H=H_len), {layer2});
	return n.add(NLAYER(name+"res", Eltwise, K=C, H=H_len, N=2), {prev, layer3});
}

static Network gen_convnext(len_t depths[], len_t dims[], int num_stages=4){
	Network n;
	InputData input("input", fmap_shape(3,224));

	lid_t prev = n.add(NLAYER("stem", Conv, C=3, K=dims[0], H=56, R=4, sH=4), {}, 0, {input});

	for(int stage=0; stage<num_stages; ++stage){
		len_t C = dims[stage];
		len_t H_len = 56 >> stage;
		for(len_t blk=0; blk<depths[stage]; ++blk){
			prev = add_convnext_block(n, prev, stage, blk, C, H_len);
		}
		if(stage < num_stages-1){
			prev = n.add(NLAYER("down_"+std::to_string(stage), Conv, C=C, K=dims[stage+1], H=H_len/2, sH=2), {prev});
		}
	}
	n.add(NLAYER("pool", Pooling, K=dims[num_stages-1], H=1, R=7), {prev});
	n.add(NLAYER("fc", FC, C=dims[num_stages-1], K=1000));
	return n;
}

static len_t depths_tiny[] = {3,3,9,3};
static len_t dims_tiny[] = {96,192,384,768};

static len_t depths_small[] = {3,3,27,3};
static len_t dims_small[] = {96,192,384,768};

static len_t depths_base[] = {3,3,27,3};
static len_t dims_base[] = {128,256,512,1024};

const Network convnext_tiny = gen_convnext(depths_tiny, dims_tiny);
const Network convnext_small = gen_convnext(depths_small, dims_small);
const Network convnext_base = gen_convnext(depths_base, dims_base);