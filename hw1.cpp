#define IMAGE_FILE  "road.png"


#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <iterator>
#include <algorithm>
#include <vector>
#include <arpa/inet.h>

using namespace std;

int length;
unsigned int width, height;
char * file_data;

unsigned int ** img;

int main(){
    ifstream file(IMAGE_FILE, ios::binary);
    istream_iterator<unsigned char>begin(file), end;
    vector<unsigned char> buffer(begin,end);
    copy(buffer.begin(), buffer.end(), ostream_iterator<unsigned int>(cout, ","));

    file.clear();
    file.seekg(0, ios::beg);

    file.seekg(16);
    file.read((char*) &width, 4);
    file.read((char*)&height, 4);

    width = ntohl(width);
    height = ntohl(height);

    img = new unsigned int*[height];
    for(unsigned int i = 0; i < height; ++i){
        img[i] = new unsigned int[width];
    }

    int k = file.tellg();
    for (unsigned int i = 0; i < height; i++){
        for (unsigned int j = 0; j < width; j++){
            img[i][j] = buffer.at(k++);
        }
    }

    
    cout <<endl<< "Img dimensions: "<<width << ", " << height <<endl;

    for (int i = 0; i < height; i++){
        delete [] img[i];
    }
    delete [] img;

    // FILE *fp;
    // fp = fopen(IMAGE_FILE, "rb");
    // fseek(fp, 0, SEEK_END);
    // length = ftell(fp);
    // rewind(fp);
    // file_data = (char *)malloc((length+1)*sizeof(char));
    // fread(file_data, length, 1, fp);
    
}