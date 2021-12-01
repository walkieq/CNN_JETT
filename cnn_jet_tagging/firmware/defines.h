#ifndef DEFINES_H_
#define DEFINES_H_

#include "ap_int.h"
#include "ap_fixed.h"
#include "nnet_utils/nnet_types.h"

//hls-fpga-machine-learning insert numbers
#define N_INPUT_1_1 16
#define N_INPUT_2_1 1
#define N_OUTPUTS_2 13
#define N_FILT_2 8
#define N_OUTPUTS_4 10
#define N_FILT_4 8
#define N_OUTPUTS_6 7
#define N_FILT_6 8
#define N_LAYER_8 32
#define N_LAYER_10 16
#define N_LAYER_12 5

//hls-fpga-machine-learning insert layer-precision
typedef ap_fixed<16,6> model_default_t;
typedef nnet::array<ap_fixed<16,6>, 1*1> input_t;
typedef nnet::array<ap_fixed<16,6>, 8*1> layer2_t;
typedef ap_fixed<16,6> conv1_weight_t;
typedef ap_fixed<16,6> conv1_bias_t;
typedef ap_fixed<16,6> conv1_relu_default_t;
typedef nnet::array<ap_fixed<16,6>, 8*1> layer3_t;
typedef nnet::array<ap_fixed<16,6>, 8*1> layer4_t;
typedef ap_fixed<16,6> conv2_weight_t;
typedef ap_fixed<16,6> conv2_bias_t;
typedef ap_fixed<16,6> conv2_relu_default_t;
typedef nnet::array<ap_fixed<16,6>, 8*1> layer5_t;
typedef nnet::array<ap_fixed<16,6>, 8*1> layer6_t;
typedef ap_fixed<16,6> conv3_weight_t;
typedef ap_fixed<16,6> conv3_bias_t;
typedef ap_fixed<16,6> conv3_relu_default_t;
typedef nnet::array<ap_fixed<16,6>, 8*1> layer7_t;
typedef nnet::array<ap_fixed<16,6>, 32*1> layer8_t;
typedef ap_fixed<16,6> fc1_weight_t;
typedef ap_fixed<16,6> fc1_bias_t;
typedef ap_fixed<16,6> fc1_relu_default_t;
typedef nnet::array<ap_fixed<16,6>, 32*1> layer9_t;
typedef nnet::array<ap_fixed<16,6>, 16*1> layer10_t;
typedef ap_fixed<16,6> fc7_weight_t;
typedef ap_fixed<16,6> fc7_bias_t;
typedef ap_fixed<16,6> fc7_relu_default_t;
typedef nnet::array<ap_fixed<16,6>, 16*1> layer11_t;
typedef nnet::array<ap_fixed<16,6>, 5*1> layer12_t;
typedef ap_fixed<16,6> output_weight_t;
typedef ap_fixed<16,6> output_bias_t;
typedef ap_fixed<16,6> output_softmax_default_t;
typedef nnet::array<ap_fixed<16,6>, 5*1> result_t;

#endif
