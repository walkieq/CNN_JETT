#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#include "ap_int.h"
#include "ap_fixed.h"

#include "nnet_utils/nnet_helpers.h"
//hls-fpga-machine-learning insert includes
#include "nnet_utils/nnet_activation.h"
#include "nnet_utils/nnet_activation_stream.h"
#include "nnet_utils/nnet_conv1d.h"
#include "nnet_utils/nnet_conv1d_stream.h"
#include "nnet_utils/nnet_dense.h"
#include "nnet_utils/nnet_dense_compressed.h"
#include "nnet_utils/nnet_dense_stream.h"
 
//hls-fpga-machine-learning insert weights
#include "weights/w2.h"
#include "weights/b2.h"
#include "weights/w4.h"
#include "weights/b4.h"
#include "weights/w6.h"
#include "weights/b6.h"
#include "weights/w8.h"
#include "weights/b8.h"
#include "weights/w10.h"
#include "weights/b10.h"
#include "weights/w12.h"
#include "weights/b12.h"

//hls-fpga-machine-learning insert layer-config
// conv1
struct config2_mult : nnet::dense_config {
    static const unsigned n_in = 4;
    static const unsigned n_out = 8;
    static const unsigned reuse_factor = 1;
    static const unsigned strategy = nnet::latency;
    typedef ap_fixed<16,6> accum_t;
    typedef conv1_bias_t bias_t;
    typedef conv1_weight_t weight_t;
    template<class x_T, class y_T, class res_T>
    using product = nnet::product::mult<x_T, y_T, res_T>;
};

struct config2 : nnet::conv1d_config {
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned in_width = 16;
    static const unsigned n_chan = N_INPUT_2_1;
    static const unsigned filt_width = 4;
    static const unsigned kernel_size = filt_width;
    static const unsigned n_filt = N_FILT_2;
    static const unsigned stride_width = 1;
    static const unsigned dilation = 1;
    static const unsigned out_width = N_OUTPUTS_2;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 0;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::latency;
    static const unsigned min_width = 7;
    static const ap_uint<filt_width> pixels[min_width];
    typedef ap_fixed<16,6> accum_t;
    typedef conv1_bias_t bias_t;
    typedef conv1_weight_t weight_t;
    typedef config2_mult mult_config;
};
const ap_uint<config2::filt_width> config2::pixels[] = {1,3,7,15,14,12,8};

// conv1_relu
struct relu_config3 : nnet::activ_config {
    static const unsigned n_in = N_OUTPUTS_2*N_FILT_2;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    typedef ap_fixed<18,8> table_t;
};

// conv2
struct config4_mult : nnet::dense_config {
    static const unsigned n_in = 32;
    static const unsigned n_out = 8;
    static const unsigned reuse_factor = 1;
    static const unsigned strategy = nnet::latency;
    typedef ap_fixed<16,6> accum_t;
    typedef conv2_bias_t bias_t;
    typedef conv2_weight_t weight_t;
    template<class x_T, class y_T, class res_T>
    using product = nnet::product::mult<x_T, y_T, res_T>;
};

struct config4 : nnet::conv1d_config {
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned in_width = 13;
    static const unsigned n_chan = N_FILT_2;
    static const unsigned filt_width = 4;
    static const unsigned kernel_size = filt_width;
    static const unsigned n_filt = N_FILT_4;
    static const unsigned stride_width = 1;
    static const unsigned dilation = 1;
    static const unsigned out_width = N_OUTPUTS_4;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 0;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::latency;
    static const unsigned min_width = 7;
    static const ap_uint<filt_width> pixels[min_width];
    typedef ap_fixed<16,6> accum_t;
    typedef conv2_bias_t bias_t;
    typedef conv2_weight_t weight_t;
    typedef config4_mult mult_config;
};
const ap_uint<config4::filt_width> config4::pixels[] = {1,3,7,15,14,12,8};

// conv2_relu
struct relu_config5 : nnet::activ_config {
    static const unsigned n_in = N_OUTPUTS_4*N_FILT_4;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    typedef ap_fixed<18,8> table_t;
};

// conv3
struct config6_mult : nnet::dense_config {
    static const unsigned n_in = 32;
    static const unsigned n_out = 8;
    static const unsigned reuse_factor = 1;
    static const unsigned strategy = nnet::latency;
    typedef ap_fixed<16,6> accum_t;
    typedef conv3_bias_t bias_t;
    typedef conv3_weight_t weight_t;
    template<class x_T, class y_T, class res_T>
    using product = nnet::product::mult<x_T, y_T, res_T>;
};

struct config6 : nnet::conv1d_config {
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned in_width = 10;
    static const unsigned n_chan = N_FILT_4;
    static const unsigned filt_width = 4;
    static const unsigned kernel_size = filt_width;
    static const unsigned n_filt = N_FILT_6;
    static const unsigned stride_width = 1;
    static const unsigned dilation = 1;
    static const unsigned out_width = N_OUTPUTS_6;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 0;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::latency;
    static const unsigned min_width = 7;
    static const ap_uint<filt_width> pixels[min_width];
    typedef ap_fixed<16,6> accum_t;
    typedef conv3_bias_t bias_t;
    typedef conv3_weight_t weight_t;
    typedef config6_mult mult_config;
};
const ap_uint<config6::filt_width> config6::pixels[] = {1,3,7,15,14,12,8};

// conv3_relu
struct relu_config7 : nnet::activ_config {
    static const unsigned n_in = N_OUTPUTS_6*N_FILT_6;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    typedef ap_fixed<18,8> table_t;
};

// fc1
struct config8 : nnet::dense_config {
    static const unsigned n_in = N_OUTPUTS_6*N_FILT_6;
    static const unsigned n_out = N_LAYER_8;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned strategy = nnet::latency;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 0;
    static const unsigned n_nonzeros = 1792;
    static const bool store_weights_in_bram = false;
    typedef ap_fixed<16,6> accum_t;
    typedef fc1_bias_t bias_t;
    typedef fc1_weight_t weight_t;
    typedef ap_uint<1> index_t;
    template<class x_T, class y_T, class res_T>
    using product = nnet::product::mult<x_T, y_T, res_T>;
};

// fc1_relu
struct relu_config9 : nnet::activ_config {
    static const unsigned n_in = N_LAYER_8;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    typedef ap_fixed<18,8> table_t;
};

// fc7
struct config10 : nnet::dense_config {
    static const unsigned n_in = N_LAYER_8;
    static const unsigned n_out = N_LAYER_10;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned strategy = nnet::latency;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 0;
    static const unsigned n_nonzeros = 512;
    static const bool store_weights_in_bram = false;
    typedef ap_fixed<16,6> accum_t;
    typedef fc7_bias_t bias_t;
    typedef fc7_weight_t weight_t;
    typedef ap_uint<1> index_t;
    template<class x_T, class y_T, class res_T>
    using product = nnet::product::mult<x_T, y_T, res_T>;
};

// fc7_relu
struct relu_config11 : nnet::activ_config {
    static const unsigned n_in = N_LAYER_10;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    typedef ap_fixed<18,8> table_t;
};

// output
struct config12 : nnet::dense_config {
    static const unsigned n_in = N_LAYER_10;
    static const unsigned n_out = N_LAYER_12;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned strategy = nnet::latency;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 0;
    static const unsigned n_nonzeros = 80;
    static const bool store_weights_in_bram = false;
    typedef ap_fixed<16,6> accum_t;
    typedef output_bias_t bias_t;
    typedef output_weight_t weight_t;
    typedef ap_uint<1> index_t;
    template<class x_T, class y_T, class res_T>
    using product = nnet::product::mult<x_T, y_T, res_T>;
};

// output_softmax
struct softmax_config13 : nnet::activ_config {
    static const unsigned n_in = N_LAYER_12;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    static const nnet::softmax_implementation implementation = nnet::softmax_implementation::latency;
    typedef ap_fixed<18,8> exp_table_t;
    typedef ap_fixed<18,8> inv_table_t;
};


#endif
