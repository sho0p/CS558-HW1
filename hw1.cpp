#define IMAGE_FILE  "road.png"


#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <iterator>
#include <algorithm>
#include <vector>

using namespace std;

int length;
char * file_data;

int main(){
    ifstream file(IMAGE_FILE, ios::binary);
    istream_iterator<unsigned char>begin(file), end;
    vector<unsigned char> buffer(begin,end);
    copy(buffer.begin(), buffer.end(), ostream_iterator<unsigned int>(cout, ","));

    

    // FILE *fp;
    // fp = fopen(IMAGE_FILE, "rb");
    // fseek(fp, 0, SEEK_END);
    // length = ftell(fp);
    // rewind(fp);
    // file_data = (char *)malloc((length+1)*sizeof(char));
    // fread(file_data, length, 1, fp);
    
}