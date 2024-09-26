#include <iostream>
#include <stdio.h>
#include <string>
#include <stdlib.h>
#include <png.h>

using namespace std;

#define IMAGE_HEIGHT 250
#define IMAGE_WIDTH 300
#define IMAGE_RED_COLOR 0
#define IMAGE_GREEN_COLOR 255
#define IMAGE_BLUE_COLOR 0
#define IMAGE_ALPHA_CHANNEL 255

class ComplexNumber{
    double r;
    double c;

    public:
    ComplexNumber(){
        r=0;
        c=0;
    }

    public:
    ComplexNumber(double r0,double c0){
        r=r0;
        c=c0;
    }

    public:
    void square(){
        double r_old=r;
        double c_old=c;
        r=r_old*r_old-c_old*c_old;
        c=2*r_old*c_old;
    }

    public:
    void sum(ComplexNumber cn){
        r+=cn.r;
        c+=cn.c;
    }

    public:
    void sum(double r0,double c0){
        r+=r0;
        c+=c0;
    }

    public:
    void print_values(){
        //cout<<r<<" + "<<c<<"i"<<std::endl;
    }

    public:
    double abs(){
        return r*r+c*c;
    }

    public:
    bool mandelbrot_converge(int diverge_threshold,int converge_threshold){
        double r0=r;
        double c0=c;
        double a0=abs();
        square();
        sum(r0,c0);
        double a1=abs();
        square();
        sum(r0,c0);
        double a2=abs();
        while(true){
            if(a2-a1>a1-a0){
                diverge_threshold--;
            }else{
                converge_threshold--;
            }
            if(diverge_threshold<=0){
                return false;
            }
            if(converge_threshold<=0){
                return true;
            }
            square();
            sum(r0,c0);
            a0=a1;
            a1=a2;
            a2=abs();
        }
    }

};






void write_png_image()
{
    png_byte** row_pointers; // pointer to image bytes
    FILE* fp; // file for image

    do // one time do-while to properly free memory and close file after error
    {
        row_pointers = (png_byte**)malloc(sizeof(png_byte*) * IMAGE_HEIGHT);
        if (!row_pointers)
        {
            printf("Allocation failed\n");
            break;
        }
        for (int i = 0; i < IMAGE_HEIGHT; i++)
        {
            row_pointers[i] = (png_byte*)malloc(4*IMAGE_WIDTH);
            if (!row_pointers[i])
            {
                printf("Allocation failed\n");
                break;
            }
        }
        // fill image with color
        for (int y = 0; y < IMAGE_HEIGHT; y++)
        {
            for (int x = 0; x < IMAGE_WIDTH*4; x+=4)
            {
                row_pointers[y][x] = IMAGE_RED_COLOR; //r
                row_pointers[y][x + 1] = IMAGE_GREEN_COLOR; //g
                row_pointers[y][x + 2] = IMAGE_BLUE_COLOR; //b
                row_pointers[y][x + 3] = IMAGE_ALPHA_CHANNEL; //a
            }
        }
        //printf("%d %d %d %d\n", row_pointers[0][0], row_pointers[0][1], row_pointers[0][2], row_pointers[0][3]);

        fp = fopen("picture.png", "wb"); //create file for output
        if (!fp)
        {
            printf("Open file failed\n");
            break;
        }
        png_struct* png = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL); //create structure for write
        if (!png)
        {
            printf("Create write struct failed\n");
            break;
        }
        png_infop info = png_create_info_struct(png); // create info structure
        if (!info)
        {
            printf("Create info struct failed\n");
            break;
        }
        if (setjmp(png_jmpbuf(png))) // this is some routine for errors?
        {
            printf("setjmp failed\n");
        }
        png_init_io(png, fp); //initialize file output
        png_set_IHDR( //set image properties
            png, //pointer to png_struct
            info, //pointer to info_struct
            IMAGE_WIDTH, //image width
            IMAGE_HEIGHT, //image height
            8, //color depth
            PNG_COLOR_TYPE_RGBA, //color type
            PNG_INTERLACE_NONE, //interlace type
            PNG_COMPRESSION_TYPE_DEFAULT, //compression type
            PNG_FILTER_TYPE_DEFAULT //filter type
            );
        png_write_info(png, info); //write png image information to file
        png_write_image(png, row_pointers); //the thing we gathered here for
        png_write_end(png, NULL);
        //printf("Image was created successfully\nCheck %s file\n", filename);
    } while(0);
    //close file
    if (fp)
    {
        fclose(fp);
    }
    //free allocated memory
    for (int i = 0; i < IMAGE_HEIGHT; i++)
    {
        if (row_pointers[i])
        {
            free(row_pointers[i]);
        }
    }
    if (row_pointers)
    {
        free(row_pointers);
    }
}



int main(int argc, char* argv[]) { 
    //double re_min=-1;
    //double re_max=1;
    //double cx_min=-1;
    //double cx_max=1;
    //double ndiv=10;
//
    //double d_re=(re_max-re_min)/ndiv;
    //double d_cx=(cx_max-cx_min)/ndiv;
//
//
    //for(int i=0;i<ndiv;i++){
    //    for(int j=0;j<ndiv;j++){
    //        double re=re_min+i*d_re;
    //        double cx=cx_min+j*d_cx;
    //        ComplexNumber c(re,cx);
    //        cout<<c.mandelbrot_converge(5,20)<<endl;
    //    }
    //}


    //write_png_image();

    return 0;

};