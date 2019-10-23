#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
using namespace std;

static string dict_file = "sex-averaged.rmap";

struct gene_pos {
  int chr;
  long pos;
};

void find_distances(vector<gene_pos>& targets, vector<float>& distances){
  assert (targets.size() >= 1);
  int chr_target = targets[0].chr;
  long prev_pos = 0;
  float accum_dist = 0.;
  ifstream dicFile;
  dicFile.open(dict_file);
  int line_chr, line_seq_bin;
  long line_pos;
  float line_dist;
  string buf;
  int vector_pos = 0;
  while (getline(dicFile, buf)){
    assert (sscanf(buf.c_str(), "chr%d\t%ld\t%d\t%f", &line_chr, &line_pos, &line_seq_bin, &line_dist) == 4);
    if (line_chr > chr_target) break;
    if (line_chr == chr_target) {
      while (true){
        if (vector_pos == targets.size()) {dicFile.close();return;}
        if (targets[vector_pos].pos >= line_pos) break;
        distances.push_back(accum_dist);
        vector_pos++; 
      }
      accum_dist += line_dist;
    }
  }
  int pad_length = targets.size() - distances.size();
  for (int i=0; i<pad_length; i++){
    distances.push_back(accum_dist);
  }
  dicFile.close();
  return;
}

int main(int argc, char* argv[]){
  assert (argc == 2);
  string path(argv[1]);
  ifstream inFile;
  inFile.open(path);
  if (!inFile) {
      cout << "Unable to open file";
      exit(1);
  }
  vector<gene_pos> targets;
  vector<float> distances;
  string buf;
  while (getline(inFile, buf)){
    size_t splitter = buf.find_first_of(' ');
    string chr = buf.substr(0, splitter);
    if (chr == "X") chr = "23";
    if (chr == "Y") chr = "24";
    int c = stol(chr);
    long pos = stol(buf.substr(splitter+1, buf.size()-splitter-1));
    gene_pos p = {c, pos};
    targets.push_back(p);
  }
  inFile.close();
  find_distances(targets, distances);
  string out_path = path + ".out";
  filebuf fb;
  fb.open(out_path, std::ios::out);
  ostream os(&fb);
  for (float dis: distances) 
    os << setw(11) << setprecision(10) << dis << endl;
  return 0;
}
