#include "Part.h"

#include <stdio.h>
#include <cstring>
#include <string>
#include <algorithm>
#include <iostream>
#include <cmath>

using namespace std;

Node::Node()
{
    x = y = z = 0;
}

Node::Node(double _x, double _y, double _z)
{
    x = _x; y = _y; z = _z;
}

double Node::len() const
{
    return sqrt(x * x + y * y + z * z);
}

Element::Element()
{
    size = 0; node_id.clear();
}

Element::Element(int _size, const vector<int> &_node_id)
{
    size = _size; 
    for (int i = 0; i < (int)_node_id.size(); ++i)
    node_id.push_back(_node_id[i]);
}

vector<int> Element::get_list()
{
    return node_id;
}

int Element::contains(int x)
{
    for (int i = 0; i < (int)node_id.size(); ++i) if (node_id[i] == x) return 1;
    return 0;
}

Part::Part()
{
    num_node = num_element = 0;
    element_set.clear();
}

Part::Part(string path, string name)
{
    part_name = name.c_str(); num_node = num_element = 0;
    FILE *fp = fopen(path.c_str(), "r");
    char buf[1005]; int flag = 0;
    while (fgets(buf, 1000, fp) != NULL)
    {
        int buf_size = strlen(buf);
        if (strstr(buf, "*Part, name=") != NULL && strstr(buf, name.c_str()) != NULL) flag = 1;
        else if (flag == 1 && strstr(buf, "*Node") != NULL) flag = 2;
        else if (flag == 2 && strstr(buf, "*Element") != NULL) flag = 3;
        else if (flag == 3 && strstr(buf, "*") != NULL) break;
        else if (flag == 2 && strstr(buf, "*") == NULL)
        {
            vector<double> tmpv; 
            int j = 0;
            for (int i = 0; i < buf_size; i = j)
            if (buf[i] == '-' || (buf[i] >= '0' && buf[i] <= '9'))
            {
                string a = "";
                for (j = i; j < buf_size && (buf[j] == '-' || buf[j] == 'e' || buf[j] == '.' || (buf[j] >= '0' && buf[j] <= '9')); ++j) a += buf[j];
                tmpv.push_back(float_change(a));
            }
            else j = i + 1; 
            ++num_node;
            node_map[(int)tmpv[0]] = Node(tmpv[1], tmpv[2], tmpv[3]);
        }
        else if (flag == 3 && strstr(buf, "*") == NULL)
        {
            vector<double> tmpv; 
            vector<int> tmpi;
            int j = 0;
            for (int i = 0; i < buf_size; i = j)
            if (buf[i] == '-' || (buf[i] >= '0' && buf[i] <= '9'))
            {
                string a = "";
                for (j = i; j < buf_size && (buf[j] == '-' || buf[j] == 'e' || buf[j] == '.' || (buf[j] >= '0' && buf[j] <= '9')); ++j) a += buf[j];
                tmpv.push_back(float_change(a));
            }
            else j = i + 1;
            for (int i = 1; i < tmpv.size(); ++i) tmpi.push_back((int)tmpv[i]);
            Element new_element = Element(tmpi.size(), tmpi);
            element_set[(int)tmpv[0]] = new_element; ++num_element;
            for (int i = 0; i < (int)tmpi.size(); ++i) element_map[tmpi[i]].push_back(new_element);
        }
    }
    fprintf(stderr, "part_name = %s, total_node = %d, tot_element = %d\n", name.c_str(), num_node, num_element);
    fclose(fp);
}

double Part::float_change(string c)
{
    int len = c.length(); 
    double a = 0, af = 1, ap = 0;
    double b = 0, bf = 1, bp = 0;
    int flag = 0;
    for (int i = 0; i < len; ++i)
    if (c[i] >= '0' && c[i] <= '9')
    {
        if (flag == 0) a = a * 10 + c[i] - '0', ap *= 10;
        else           b = b * 10 + c[i] - '0', bp *= 10;
    }
    else if (c[i] == '.')
    {
        if (flag == 0) ap = 1; else bp = 1;
    }
    else if (c[i] == '-')
    {
        if (flag == 0) af = -1; else bf = -1;
    }
    else if (c[i] == 'e')
    {
        flag = 1;
    }
    else fprintf(stderr, "Error in change number\n");
    if (ap > 0.5) a /= ap; if (bp > 0.5) b /= bp;
    a *= af; b *= bf;
    while (b > 0.5) a *= 10, b -= 1;
    while (b < -0.5) a /= 10, b += 1;
    return a;
}

Node Part::get_node(int id)
{
    return node_map[id];
}

vector<Element> Part::get_element(int id)
{
    return element_map[id];
}

void Part::output_node(int id)
{
    Node tmp = get_node(id);
    fprintf(stderr, "%d %lf %lf %lf\n", id, tmp.x, tmp.y, tmp.z);
}

void Part::node_print(string path_name)
{
   freopen(path_name.c_str(), "w", stdout);
   //printf("ply\nformat ascii 1.0\nelement vertex %d\nproperty double x\nproperty double y\nproperty double z\nend_header\n", num_node);
   for (map<int, Node>::iterator it = node_map.begin(); it != node_map.end(); ++it)
   printf("%.10lf %.10lf %.10lf\n", it->second.x, it->second.y, it->second.z);
}

void Part::try_generate_face()
{
    Node_Next.reserve(num_node + 1);
    Node_Fa.reserve(num_node + 1);

    for (int i = 1; i <= num_node; ++i) Node_Fa[i] = Node_Next[i] = 0;
    for (map<int, Element>::iterator it = element_set.begin(); it != element_set.end(); ++it)
    {
        Element *p = &it->second; double a = calc_z(p->node_id);
        vector<int> new_id_1;
        new_id_1.push_back(p->node_id[0]); new_id_1.push_back(p->node_id[4]); new_id_1.push_back(p->node_id[7]); new_id_1.push_back(p->node_id[3]);
        new_id_1.push_back(p->node_id[1]); new_id_1.push_back(p->node_id[5]); new_id_1.push_back(p->node_id[6]); new_id_1.push_back(p->node_id[2]);
        double b = calc_z(new_id_1);        
        vector<int> new_id_2;
        new_id_2.push_back(p->node_id[0]); new_id_2.push_back(p->node_id[4]); new_id_2.push_back(p->node_id[5]); new_id_2.push_back(p->node_id[1]);
        new_id_2.push_back(p->node_id[3]); new_id_2.push_back(p->node_id[7]); new_id_2.push_back(p->node_id[6]); new_id_2.push_back(p->node_id[2]);
        double c = calc_z(new_id_2);
        if (b < c && b < a) p->node_id = new_id_1;
        else if (c < a && c < b) p->node_id = new_id_2;       
 //       printf("%.10lf %.10lf %.10lf\n", a, b, c);
        if (not_good(p->node_id)) for (int i = 0; i < 4; ++i) swap(p->node_id[i], p->node_id[i + 4]);
 //       for (int i = 0; i < 8; ++i) printf("%d ", p->node_id[i]);
//          for (int i = 0; i < 8; ++i) printf("%d ", p->node_id[i]);
//        puts("");
        for (int i = 0; i < 4; ++i) 
        {
            int a = p->node_id[i], b = p->node_id[i + 4];
            if ((Node_Next[a] && Node_Next[a] != b)|| (Node_Fa[b] && Node_Fa[b] != a)) 
            {
                printf("%d %d %d %d GGG\n", Node_Next[a], Node_Fa[b], a, b);
                for (int i = 0; i < 8; ++i) printf("%d ", p->node_id[i]);
                puts("");
            }
            Node_Next[a] = b;
            Node_Fa[b] = a;
        }
    }
}

int Part::not_good(vector<int> &v)
{
    double za = 0, zb = 0;
    for (int i = 0; i < 4; ++i)
    {
        za += get_node(v[i]).x;
    }
    for (int i = 4; i < 8; ++i)
    {
        zb += get_node(v[i]).x;
    }
    return za > zb;
}

double Part::calc_z(vector<int>& v)
{
    double zmax = -1e9, zmin = 1e9, ans = 0;
    for (int i = 0; i < 4; ++i)
    {
        zmax = max(zmax, get_node(v[i]).x);
        zmin = min(zmin, get_node(v[i]).x);
    }
    ans += zmax - zmin;
    zmax = -1e9, zmin = 1e9;
    for (int i = 5; i < 8; ++i)
    {
        zmax = max(zmax, get_node(v[i]).x);
        zmin = min(zmin, get_node(v[i]).x);
    }
    ans += zmax - zmin;
    return ans;
}

void Part::output_path(Node ref_point, Node turn_to_point, string filename)
{
    freopen(filename.c_str(), "w", stdout);
    double mini_dist = 1e20; int node_id = -1;
    Node pt;
    for (map<int, Node>::iterator it = node_map.begin(); it != node_map.end(); ++it)
    //if (it->second.x <= ref_point.x)
    {
        double now_dist = (ref_point - (it->second)).len();
        if (now_dist < mini_dist) mini_dist = now_dist, node_id = it->first, pt = it->second;
    }
    //vector<Node> ans; 
    for (int i = node_id; i; i = Node_Next[i]) printf("%.10lf %.10lf %.10lf\n", node_map[i].x + turn_to_point.x - pt.x, 
    node_map[i].y + turn_to_point.y - pt.y, node_map[i].z + turn_to_point.z - pt.z);
    //double len = 0;
    //for (int i = 0; i < (int)ans.size() - 1; ++i)
   	//	len += (ans[i] - ans[i + 1]).len();
   	//fprintf(stderr,"%.10lf\n", len);
}

void Part::output_face(char* name)
{
    vector<Node> ans;
    map<pair<int, int>, int> cross_map;
    map<int, vector<int> > edge_map;
    for (map<int, Element>::iterator it = element_set.begin(); it != element_set.end(); ++it)
    {
        Element *p = &it->second;
        if (Node_Fa[p->node_id[0]]) continue;
        for (int i = 0;i < 4; ++i)
        {
            cross_map[make_pair(p->node_id[i], p->node_id[(i + 2) & 3])] = 1;
            edge_map[p->node_id[i]].push_back(p->node_id[(i + 1) & 3]);
            edge_map[p->node_id[(i + 1) & 3]].push_back(p->node_id[i]);
        }
    }
    double dis = -1; int idx, idy;
    for (int i = 1; i <= num_node; ++i)
        if (!Node_Fa[i])
        {
            int j;
            for (j = i; Node_Next[j]; j = Node_Next[j]);
            double tmp_dis = (get_node(j) - get_node(i)).len();
            if (tmp_dis > dis) dis = tmp_dis, idx = i, idy = j; 
        }
    ans.push_back(get_node(idx));
    ans.push_back(get_node(idy));
    for (map<int, vector<int> >::iterator it = edge_map.begin(); it != edge_map.end(); ++it)
    {
        int nowid = it->first; 
        ans.push_back(get_node(nowid));
        Node ref_point = get_node(nowid);
        int tmp[50]; tmp[0] = 0;
        for (int i = 0 ; i < it->second.size(); ++i) tmp[++tmp[0]] = it->second[i];
        sort(tmp + 1, tmp + tmp[0] + 1);
        tmp[0] = unique(tmp + 1, tmp + tmp[0] + 1) - (tmp + 1);
        for (int i = 1; i <= tmp[0]; ++i)
         for (int j = i + 1; j <= tmp[0]; ++j)
            {
                Node pa = get_node(tmp[i]) - ref_point;
                Node pb = get_node(tmp[j]) - ref_point;
                if (atan2(pa.y, pa.x) > atan2(pb.y, pb.x)) swap(tmp[i], tmp[j]);
            }
        int flag = 0;
        fprintf(stderr, "%.10lf %.10lf %.10lf %d\n", ref_point.x, ref_point.y, ref_point.z, (int)tmp[0]);
        tmp[tmp[0] + 1] = tmp[1];
        for  (int i = 1; i <= tmp[0]; ++i)
            if (cross_map.find(make_pair(tmp[i], tmp[i + 1])) == cross_map.end())
            {
                flag = 1; break;
            }
        if (flag) 
        {
            for (int i = Node_Next[nowid]; i; i = Node_Next[i]) ans.push_back(get_node(i));
        }
        else
        {
            for (int i = Node_Next[nowid]; i; i = Node_Next[i]) if (!Node_Next[i]) ans.push_back(get_node(i));
        }
    }
//    string m = name; m = "../est/" + m + ".txt";
//   freopen(m.c_str(), "w", stdout);
//   printf("ply\nformat ascii 1.0\nelement vertex %d\nproperty double x\nproperty double y\nproperty double z\nend_header\n", (int)ans.size());
   double len = 0;
   for (int i = 0; i < (int)ans.size() - 1; ++i)
   	len += (ans[i] - ans[i + 1]).len();
   	fprintf(stderr,"%.10lf\n", len);
	//   printf("%.10lf %.10lf %.10lf\n", ans[i].x, ans[i].y, ans[i].z);
}
