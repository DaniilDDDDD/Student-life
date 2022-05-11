from parsers import parse
from distance_counters import count_distance, bfs


def main():
    data = parse('../datasets/Wiki-Vote.txt', True)

    landmarks = [7517, 1106, 5558, 4284, 6216, 5915, 4349, 1375, 6936, 1442, 5427, 4433, 3093, 700, 1794, 5950, 7480, 4315, 396, 8012, 522, 1996, 2522, 5888, 236, 3646, 2928, 1596, 1882, 6727, 4870, 8289, 1868, 3010, 811, 4591, 1005, 899, 6399, 2443, 1563, 8277, 2902, 534, 18, 1470, 1388, 5428, 7833, 5079, 7223, 7948, 6551, 79, 2139, 1652, 3694, 6928, 332, 4742, 6274, 2543, 841, 3526, 5733, 7567, 5465, 8015, 805, 1141, 1436, 4049, 7318, 7821, 2512, 362, 3973, 2862, 2690, 3487, 742, 6826, 4036, 3726, 4887, 7933, 5956, 5571, 8103, 5377, 4504, 5269, 4250, 92, 5479, 7349, 7189, 4776, 2267, 5729, 6896, 3709, 1115, 6070, 924, 2848, 1518, 6991, 6696, 6451, 121, 8226, 1215, 2496, 6902, 2921, 6572, 2233, 1458, 1841, 8140, 8038, 6273, 3214, 6472, 1094, 5314, 7605, 189, 5784, 5095, 2042, 6359, 2414, 128, 5177, 1441, 8104, 3581, 90, 208, 5271, 3757, 4006, 4110, 6487, 3486, 8033, 1016, 3472, 5722, 7927, 7113, 912, 4404, 2287, 7045, 3856, 1254, 3128, 5031, 5625, 7534, 8225, 3090, 3834, 6962, 8172, 4718, 4774, 3536, 752, 5678, 188, 3844, 2975, 4352, 3426, 4085, 831, 4580, 5224, 3182, 969, 4083, 114, 1431, 3266, 2013, 2131, 5380, 7125, 6230, 4452, 7908, 6940, 1691, 7001, 1260, 2507, 5264, 5276, 5220, 792, 6577, 2716, 595, 1654, 118, 1386, 7, 4073, 2236, 2040, 83, 4143, 7494, 3567, 7773, 2879, 664, 7564, 8139, 3080, 8194, 2079, 3, 4222, 4030, 3943, 6118, 3659, 3617, 166, 5435, 7049, 2112, 1435, 883, 2480, 134, 3933, 1831, 7159, 2315, 8147, 1925, 3797, 240, 44, 1406, 1401, 4948, 4688, 794, 4621, 7118, 1081, 3545, 5935, 1685, 3655, 4739, 4832, 1584, 5382, 3960, 3886, 3341, 4406, 4236, 4611, 3228, 3481, 7775, 8045, 4883, 7320, 5993, 927, 2688, 2611, 3623, 4244, 4847, 4891, 7615, 6732, 4850, 6494, 8270, 6912, 1409, 945, 5565, 4885, 1301, 6835, 6625, 754, 5521, 7103, 4908, 7720, 2823, 4900, 2950, 2718, 4364, 4142, 6541, 2227, 6055, 786, 2914, 7523, 1480, 6423, 2387, 5320, 4825, 1181, 2855, 3512, 7940, 4594, 5501, 4180, 4047, 5741, 7195, 4912, 7704, 574, 2274, 3632, 5343, 3850, 7829, 4371, 3198, 5907, 1372, 2840, 4960, 4728, 3135, 3280, 5438, 4291, 1821, 7133, 8149, 5615, 8080, 6675, 455, 4399, 783, 5370, 1293, 7245, 3552, 6506, 6859, 7845, 886, 1578, 4394, 7831, 2222, 8068, 7537, 5604, 1889, 6165, 3656, 4067, 7801, 832, 6609, 3878, 4357, 6547, 3113, 6674, 6367, 2313, 6452, 1405, 7765, 4369, 5599, 4896, 657, 6516, 5973, 7111, 1647, 5934, 292, 3991, 5240, 5704, 6183, 7861, 1139, 4567, 1107, 5337, 4316, 7071, 6280, 995, 6034, 3312, 5609, 547, 125, 4320, 2609, 1040, 344, 1657, 652, 4805, 307, 2118, 1874, 4640, 8252, 8085, 1087, 6612, 5183, 2265, 2299, 3662, 6515, 2521, 7150, 194, 6275, 1474, 2020, 335, 1632, 7171, 6106, 1271, 7194, 1030, 3591, 910, 4744, 823, 3569, 2155, 5580, 62, 5850, 7856, 4695, 3412, 2584, 4807, 5660, 1142, 741, 6396, 434, 2399, 889, 2731, 3593, 480, 5507, 5064, 7515, 4204, 2895, 1205, 6408, 3383, 4817, 572, 7791, 4490, 2614, 4104, 264, 6288, 6395, 869, 3723, 2136, 2872, 1681, 4131, 3863, 4601, 6492, 3980, 3727, 1707, 4245, 7870, 2477, 7402, 5712, 2586, 1502, 4046, 4380, 1829, 5730, 3942, 7129, 6090, 2018, 3350, 4844, 3449, 7593, 6685, 3223, 6657, 6067, 4460, 621, 2446, 3467, 3817, 3406, 3974, 2887, 2143, 5910, 424, 1694, 1645, 4947, 185, 7767, 681, 1231, 207, 444, 3747, 5612, 4254, 4257, 1666, 999, 2453, 190, 7865, 267, 1077, 1676, 3091, 1737, 4921, 3811, 2530, 2784, 2012, 5965, 1959, 7586, 2099, 3110, 8256, 5080, 336, 5299, 5091, 7637, 3162, 5661, 800, 4586, 4615, 3957, 8054, 193, 1034, 3858, 8181, 4344, 6192, 7155, 145, 2711, 2868, 8157, 1262, 376, 4879, 7220, 4070, 3713, 1099, 3610, 1019, 312, 2930, 1326, 1923, 865, 5813, 103, 1059, 5013, 5367, 7982, 88, 1962, 3824, 5110, 2081, 6819, 6776, 5424, 2129, 4603, 4658, 2861, 7422, 5213, 7470, 4897, 5446, 1609, 7405, 1063, 2072, 5184, 7321, 7271, 2064, 5374, 6079, 7266, 6379, 5053, 7759, 3542, 4608, 3277, 31, 6627, 5531, 3750, 7901, 1130, 51, 4675, 1958, 6779, 3770, 5243, 2351, 1618, 8202, 2000, 2952, 6002, 6652, 3402, 2149, 8010, 6391, 3619, 6801, 3003, 137, 4497, 251, 5209, 7782, 6429, 3702, 4607, 1025, 6134, 1677, 6162, 7212, 5990, 1770, 7669, 7034, 2678, 5322, 3652, 461, 2150, 5442, 6688, 6866, 7187, 6821, 1873, 5480, 4383, 3558, 853, 3416, 1303, 3578]

    for landmark in landmarks:
        count_distance(landmark, data, h=1, full=True)


    print(data[1965])
    print(bfs(6321, 4953, data))

if __name__ == '__main__':
    main()


# Source: 5375
# Stock: 1608
# Distance BFS:-1
# Distance Landmarks: 2.449489742783178