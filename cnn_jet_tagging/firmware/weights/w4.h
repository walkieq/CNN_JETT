//Numpy array shape [4, 8, 8]
//Min -0.781208932400
//Max 0.566717147827
//Number of zeros 0

#ifndef W4_H_
#define W4_H_

#ifndef __SYNTHESIS__
conv2_weight_t w4[256];
#else
conv2_weight_t w4[256] = {0.2514018714, 0.2447816283, 0.1112457216, -0.0769372061, -0.3702706695, -0.0934742764, -0.2280732393, 0.3959316909, 0.0379886031, -0.1109261662, 0.0567800812, -0.0524185039, 0.2655289471, 0.0781973153, -0.0722660124, 0.1257738620, -0.0720888004, 0.2269233763, 0.0530592315, -0.2393786609, -0.2900628746, -0.2310772538, 0.0344515294, 0.3375115395, -0.2840279341, -0.0992209390, 0.1474395245, 0.5335426331, -0.0882360637, -0.2212546915, -0.1084200814, -0.1758168638, 0.1664694250, 0.2852589488, 0.3203920126, -0.0743284523, -0.0619250797, -0.3469580412, 0.3525050581, 0.0311020520, 0.1709516943, 0.2185856253, 0.0965042040, 0.1383184642, -0.0699833333, -0.2822987735, 0.4060441256, -0.1545138210, 0.2225687504, 0.1284384578, 0.0010595315, -0.0893936381, 0.1345991939, -0.2197463810, 0.2377257794, 0.0666480362, 0.1581097841, 0.1877506971, 0.2634656131, -0.3231972158, 0.0962948203, -0.0535353124, 0.0824939907, 0.0186171811, 0.2518333197, 0.2670641840, 0.2907479405, 0.1244665682, -0.0487589724, 0.1965641528, -0.0940633267, 0.3455851078, 0.3652138710, -0.1287825257, 0.1807590425, 0.3421800733, 0.2362976074, -0.0767071396, -0.1029618010, -0.2021623105, 0.0402926542, -0.0293436646, -0.1344103068, 0.1868443042, 0.2091411203, -0.1600753963, 0.1061452851, 0.1307630837, -0.1499515623, 0.2082299888, -0.3287352026, 0.1842200607, -0.0466941111, 0.2241548598, -0.1520527303, -0.2209963650, -0.2168841809, -0.1549905092, -0.0854456574, 0.2249691486, -0.0632705614, 0.0621389076, 0.2409942150, -0.1504758447, -0.2002912015, 0.0256733596, -0.0142621314, -0.2011976838, 0.0461855009, -0.1269146502, 0.2741902471, -0.4620937407, -0.3251614273, -0.0225210767, -0.1787219793, 0.2468143851, 0.2536348104, -0.4101046622, 0.3588250875, -0.1580471396, 0.0430951826, -0.0913320184, -0.1620303839, 0.0136821382, 0.0164716840, 0.1633810848, 0.2161803246, -0.0775387734, 0.3247459829, 0.1038408503, 0.1965635121, -0.1466112584, 0.0802017376, -0.0420186669, 0.3087007403, -0.1565476060, -0.0659883544, 0.1185540706, 0.0470266603, 0.0277502481, 0.1040879562, -0.0047359695, -0.1324672699, 0.0872520432, 0.0580481216, -0.0880695656, 0.1164704189, -0.1053818911, 0.1755865812, -0.0513636358, 0.1785973310, 0.0787985921, 0.2959536612, 0.1610797197, -0.2205172777, -0.0571918190, -0.0120854126, 0.1579192281, 0.0067387908, 0.2047480345, 0.5262755156, -0.1672396809, -0.1522946209, -0.1106171757, 0.2636474073, 0.1932345778, -0.1454044282, -0.2351323068, -0.1554323584, 0.2293224633, 0.1477870792, -0.1172171086, 0.0529118814, 0.0999674648, 0.1652791351, 0.2082027942, -0.1829938143, 0.4019024372, -0.0697658136, -0.1844357401, 0.1575814784, -0.2814395726, 0.0003449490, -0.7326754332, 0.0856606737, 0.1282896698, 0.1842007041, 0.4053225517, 0.1101790443, -0.0608637631, 0.0058572479, 0.1359551549, 0.1507242471, -0.1971955299, 0.0159821864, 0.0823850408, -0.1181459054, 0.1891185045, -0.0849801674, -0.0147727849, 0.3183711171, -0.0665547848, -0.6420486569, -0.1749688238, -0.4755881131, 0.1595695764, -0.1384532452, 0.4390689433, 0.0709835291, -0.0581366457, -0.2930851579, 0.1148518398, -0.3920344710, -0.0569693893, -0.2994280159, -0.1228540614, 0.3078492880, -0.0958834961, -0.2729632556, 0.0249916166, -0.1231013387, -0.1449065059, -0.2310919613, 0.4017119110, 0.3628324270, 0.0291226786, 0.1017763242, -0.0895789340, 0.5667171478, -0.0373294987, 0.2046505362, -0.1546983868, 0.2844146192, -0.1518710703, -0.2445961833, -0.0391501859, -0.1149236932, 0.1442535371, -0.0022145335, -0.0253662355, 0.4128570259, 0.0126961758, 0.0447141565, 0.0986435041, 0.2361826748, 0.2035384327, 0.1633287370, 0.0784571320, 0.3627033830, -0.0862926021, 0.3297607899, 0.0776244849, 0.5468382239, 0.0045912345, 0.3435322642, -0.7812089324};
#endif

#endif
