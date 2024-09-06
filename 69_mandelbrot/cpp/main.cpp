#include <iostream>
#include <png.h>

using namespace std;

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
    void print_values(){
        cout<<r<<" + "<<c<<"i"<<endl;
    }
};

int main(int argc, char* argv[]) { 

    ComplexNumber c(1,1);
    c.print_values();
    return 0;

};