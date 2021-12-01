//
//    rfnoc-hls-neuralnet: Vivado HLS code for neural-net building blocks
//
//    Copyright (C) 2017 EJ Kreinar
//
//    This program is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.
//
//    This program is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public License
//    along with this program.  If not, see <http://www.gnu.org/licenses/>.
//
#include <iostream>

#include "myproject.h"
#include "parameters.h"

void myproject(
    hls::stream<input_t> &conv1_input,
    hls::stream<result_t> &layer13_out,
    unsigned short &const_size_in_1,
    unsigned short &const_size_out_1
) {

    //hls-fpga-machine-learning insert IO
    #pragma HLS INTERFACE axis port=conv1_input,layer13_out 
    #pragma HLS DATAFLOW 

    const_size_in_1 = N_INPUT_1_1*N_INPUT_2_1;
    const_size_out_1 = N_LAYER_12;

#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        //hls-fpga-machine-learning insert load weights
        nnet::load_weights_from_txt<conv1_weight_t, 32>(w2, "w2.txt");
        nnet::load_weights_from_txt<conv1_bias_t, 8>(b2, "b2.txt");
        nnet::load_weights_from_txt<conv2_weight_t, 256>(w4, "w4.txt");
        nnet::load_weights_from_txt<conv2_bias_t, 8>(b4, "b4.txt");
        nnet::load_weights_from_txt<conv3_weight_t, 256>(w6, "w6.txt");
        nnet::load_weights_from_txt<conv3_bias_t, 8>(b6, "b6.txt");
        nnet::load_weights_from_txt<fc1_weight_t, 1792>(w8, "w8.txt");
        nnet::load_weights_from_txt<fc1_bias_t, 32>(b8, "b8.txt");
        nnet::load_weights_from_txt<fc7_weight_t, 512>(w10, "w10.txt");
        nnet::load_weights_from_txt<fc7_bias_t, 16>(b10, "b10.txt");
        nnet::load_weights_from_txt<output_weight_t, 80>(w12, "w12.txt");
        nnet::load_weights_from_txt<output_bias_t, 5>(b12, "b12.txt");
        loaded_weights = true;
    }
#endif

    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    //hls-fpga-machine-learning insert layers

    hls::stream<layer2_t> layer2_out("layer2_out");
    #pragma HLS STREAM variable=layer2_out depth=13
    nnet::conv_1d_cl<input_t, layer2_t, config2>(conv1_input, layer2_out, w2, b2); // conv1

    hls::stream<layer3_t> layer3_out("layer3_out");
    #pragma HLS STREAM variable=layer3_out depth=13
    nnet::relu<layer2_t, layer3_t, relu_config3>(layer2_out, layer3_out); // conv1_relu

    hls::stream<layer4_t> layer4_out("layer4_out");
    #pragma HLS STREAM variable=layer4_out depth=10
    nnet::conv_1d_cl<layer3_t, layer4_t, config4>(layer3_out, layer4_out, w4, b4); // conv2

    hls::stream<layer5_t> layer5_out("layer5_out");
    #pragma HLS STREAM variable=layer5_out depth=10
    nnet::relu<layer4_t, layer5_t, relu_config5>(layer4_out, layer5_out); // conv2_relu

    hls::stream<layer6_t> layer6_out("layer6_out");
    #pragma HLS STREAM variable=layer6_out depth=7
    nnet::conv_1d_cl<layer5_t, layer6_t, config6>(layer5_out, layer6_out, w6, b6); // conv3

    hls::stream<layer7_t> layer7_out("layer7_out");
    #pragma HLS STREAM variable=layer7_out depth=7
    nnet::relu<layer6_t, layer7_t, relu_config7>(layer6_out, layer7_out); // conv3_relu

    hls::stream<layer8_t> layer8_out("layer8_out");
    #pragma HLS STREAM variable=layer8_out depth=1
    nnet::dense<layer7_t, layer8_t, config8>(layer7_out, layer8_out, w8, b8); // fc1

    hls::stream<layer9_t> layer9_out("layer9_out");
    #pragma HLS STREAM variable=layer9_out depth=1
    nnet::relu<layer8_t, layer9_t, relu_config9>(layer8_out, layer9_out); // fc1_relu

    hls::stream<layer10_t> layer10_out("layer10_out");
    #pragma HLS STREAM variable=layer10_out depth=1
    nnet::dense<layer9_t, layer10_t, config10>(layer9_out, layer10_out, w10, b10); // fc7

    hls::stream<layer11_t> layer11_out("layer11_out");
    #pragma HLS STREAM variable=layer11_out depth=1
    nnet::relu<layer10_t, layer11_t, relu_config11>(layer10_out, layer11_out); // fc7_relu

    hls::stream<layer12_t> layer12_out("layer12_out");
    #pragma HLS STREAM variable=layer12_out depth=1
    nnet::dense<layer11_t, layer12_t, config12>(layer11_out, layer12_out, w12, b12); // output

    nnet::softmax<layer12_t, result_t, softmax_config13>(layer12_out, layer13_out); // output_softmax

}
