#define IMAGE_FILE  "road.png"


#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <iterator>
#include <algorithm>
#include <vector>
#include <arpa/inet.h>
#include <png.h>
#include <stdarg.h>

using namespace std;

int length;
unsigned int width, height;
char * file_data;

unsigned int ** img;

png_byte color_type;
png_byte bit_depth;

png_structp png_ptr;
png_infop info_ptr;
int number_of_passes;
png_bytep * row_pointers;

void abort_(const char* s, ...){
    va_list args;
    va_start(args, s);
    vfprintf(stderr, s, args);
    fprintf(stderr, "\n");
    va_end(args);
    abort();
}

void read_png_file(char * filename){
    char header[8];

    FILE *fp = fopen(filename, "rb");
    if(!fp)
        abort_("[read_png_file] File %s could not be opened for reading", filename);
    fread(header, 1, 8, fp);
    if (png_sig_cmp((png_bytep)header, 0, 8))
        abort_("[read_png_file] file %s is not recognized as a PNG file",filename);
    
    png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);

    if(!png_ptr)
        abort_("[read_png_file] png_create_read_struct failed");
    
    info_ptr = png_create_info_struct(png_ptr);
    if(!info_ptr)
        abort_("[read_png_file] png_create_info_struct failed");
    
    if(setjmp(png_jmpbuf(png_ptr)))
        abort_("[read_png_file] error during init_io");

    png_init_io(png_ptr, fp);
    png_set_sig_bytes(png_ptr, 8);

    png_read_info(png_ptr, info_ptr);

    width = png_get_image_width(png_ptr, info_ptr);
    height = png_get_image_height(png_ptr, info_ptr);
    color_type = png_get_color_type(png_ptr, info_ptr);
    bit_depth = png_get_bit_depth(png_ptr, info_ptr);

    number_of_passes = png_set_interlace_handling(png_ptr);
    png_read_update_info(png_ptr, info_ptr);

    /*actually read the file*/
    if(setjmp(png_jmpbuf(png_ptr)))
        abort_("[read_png_file] Error during read_image");
    
    row_pointers = (png_bytep*) malloc(sizeof(png_bytep) * height);

    for(unsigned int y = 0; y < height; y++){
        row_pointers[y] = (png_byte*) malloc(png_get_rowbytes(png_ptr,info_ptr));
    }

    png_read_image(png_ptr, row_pointers);
    fclose(fp);
}   

void cat_file(void){
    if(png_get_color_type(png_ptr, info_ptr) == PNG_COLOR_TYPE_RGB)
        abort_("[process_file] input file is PNG_COLOR_TYPE_RGB but must be PNG_COLOR_TYPE_RGBA " "(lacks the alpha channel)");
    
    // if(png_get_color_type(png_ptr, info_ptr) != PNG_COLOR_TYPE_RGBA)
    //     abort_("[process_file] color_type of input file must be PNG_COLOR_TYPE_RGBA (%d) (is %d)", \
    //     PNG_COLOR_TYPE_RGBA, png_get_color_type(png_ptr, info_ptr));
    for(unsigned int y = 0; y < height; y++){
        png_byte* row = row_pointers[y];
        for(unsigned int x = 0; x < width; x++){
            png_byte * ptr = &(row[x*4]);
            printf("Pixel at position [ %d - %d ] has RGBA values: %d - %d - %d - %d\n", 
            x, y, ptr[0], ptr[1], ptr[2], ptr[3]);
        }
    }
    
}

int main(int argc, char ** argv){

    if(argc != 2){
        abort_("Usage: %s <file_in>", argv[0]);
    }
    read_png_file(argv[1]);
    cat_file();
    // FILE *fp;
    // fp = fopen(IMAGE_FILE, "rb");
    // fseek(fp, 0, SEEK_END);
    // length = ftell(fp);
    // rewind(fp);
    // file_data = (char *)malloc((length+1)*sizeof(char));
    // fread(file_data, length, 1, fp);
}