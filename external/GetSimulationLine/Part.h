#ifndef PART_H
#define PART_H

#include <vector>
#include <string>
#include <map>

using namespace std;

class Node{
    public:
    double x, y, z;
    Node();
    Node(double _x, double _y, double _z);
    double len() const;
};

inline Node operator -(const Node &a, const Node &b)
{
    return Node(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline Node operator +(const Node &a, const Node &b)
{
    return Node(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline Node operator *(double a, const Node &b)
{
    return Node(a * b.x, a * b.y, a * b.z);
}

class Element{
    public:
    vector<int> node_id;
    int size;
    Element();
    Element(int _size, const vector<int> &_node_id);
    int contains(int x);
    vector<int> get_list();
};

class Part{
    public:
    vector<int> Node_Next, Node_Fa;
    map<int, Node> node_map;
    map<int, vector<Element> > element_map;
    map<int, Element> element_set;
    int num_node, num_element;
    string part_name;
    Part();
    Part(string path, string name);
    double float_change(string c);
    Node get_node(int id);
    vector<Element> get_element(int id);
    void output_face(char* name);
    void output_node(int id);
    void try_generate_face();
    void node_print(string path_name);
    double calc_z(vector<int>& v);
    int not_good(vector<int>& v);
    void output_path(Node ref_point, Node turn_to_point, string filename);
};
#endif