#include <bits/stdc++.h>
#include <string>
#include "Part.h"

using namespace std;

string base_path, static_path, recursion_path, outp_path_1, outp_path_2, outp_path_3, file_start_points_1, file_start_points_2, base_name;

int main(int argc, char *argv[])
{
    if (argc != 3) fprintf(stderr, "需要输入数据路径!\n");
    string static_path = argv[1];
    string recursion_path = argv[2];
    //file_start_points = argv[3];
    file_start_points_1 = static_path + "feature_line_ref_0.txt";
    file_start_points_2 = static_path + "feature_line_ref_1.txt";
    base_path = recursion_path + "simulation/inp_from_odb.inp";
    outp_path_1 = recursion_path + "simulation/";
    outp_path_2 = outp_path_1;
    outp_path_1 += "simulation_line_0.txt";
    outp_path_2 += "simulation_line_1.txt";
    base_name = "STRIP";
    printf("line 1 output to %s\n", outp_path_1.c_str());
    printf("line 2 output to %s\n", outp_path_2.c_str());
    Part base_part = Part(base_path, base_name);
    base_part.try_generate_face();
    freopen(file_start_points_1.c_str(), "r", stdin);
    Node ref_point[2];
    scanf("%lf %lf %lf", &ref_point[0].x, &ref_point[0].y, &ref_point[0].z);
    freopen(file_start_points_2.c_str(), "r", stdin);
    scanf("%lf %lf %lf", &ref_point[1].x, &ref_point[1].y, &ref_point[1].z);
    base_part.output_path(ref_point[0], ref_point[0], outp_path_1);
    base_part.output_path(ref_point[1], ref_point[1], outp_path_2);
}
