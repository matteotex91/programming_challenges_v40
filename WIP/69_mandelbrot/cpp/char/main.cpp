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
    int mandelbrot_converge(int diverge_threshold,int converge_threshold){
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
                return 0;
            }
            if(converge_threshold<=0){
                return 1;
            }
            square();
            sum(r0,c0);
            a0=a1;
            a1=a2;
            a2=abs();
        }
    }

};



int main(int argc, char* argv[]) { 
    double re_min=-1;
    double re_max=1;
    double cx_min=-1;
    double cx_max=1;
    double ndiv=100;
    double d_re=(re_max-re_min)/ndiv;
    double d_cx=(cx_max-cx_min)/ndiv;
    for(int i=0;i<ndiv;i++){
        for(int j=0;j<ndiv;j++){
            double re=re_min+i*d_re;
            double cx=cx_min+j*d_cx;
            ComplexNumber c(re,cx);
            cout<<c.mandelbrot_converge(5,100);
        }
        cout<<endl;
    }

    return 0;

};