#include <bits/stdc++.h>
#include <string>
#include "Part.h"

using namespace std;

string base_path, static_path, recursion_path, outp_path_1, outp_path_2, file_start_points, base_name;

int main(int argc, char *argv[])
{
    if (argc != 2) fprintf(stderr, "需要输入数据路径!\n");
    string path = argv[1];
    //file_start_points = argv[3];
    base_path = path + "inp_from_odb.inp";
    outp_path_1 = path + "springback_strip_line.txt";
    outp_path_2 = path + "springback_strip_pc.txt";
    base_name = "STRIP";
    fprintf(stderr, "final line output to %s\n", outp_path_1.c_str());
    fprintf(stderr, "final pc output to %s\n", outp_path_2.c_str());
    Part base_part = Part(base_path, base_name);
    base_part.try_generate_face();
    base_part.output_path((Node){0, 0, 0}, (Node){0, 0, 0}, outp_path_1);
    base_part.node_print(outp_path_2);
}
