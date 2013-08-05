/*
 * File:   main.cpp
 * Author: allan
 *
 * Created on April 16, 2013, 5:03 PM
 */

#define PROFILING
#define __NO_STD_VECTOR
#define __CL_ENABLE_EXCEPTIONS

#include <cstdlib>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cmath>
#include <errno.h>
#include <sys/stat.h>
#include <inttypes.h>
//#include <cutil_inline.h>

#include <CL/cl.hpp>

using namespace std;

static unsigned long *h_data, *h_out, *h_cout, *h_dout;

class keccakOCL{

private:
	cl::Kernel kernel;
	cl::Buffer d_data, d_out, d_cout, d_dout;
	cl_int size;

public:
	keccakOCL(cl::Context& context, cl::vector<cl::Device>& devices, const char* kernelFile, int qntHash){
                int blocksize = 5;
                int width = 5 * qntHash;
                int height = 5 * qntHash;
                string buildOptions;
                {
                  char buf[256];
                  sprintf(buf,"-D BLOCK_SIZE=%d -D WIDTH=%d ",
                    blocksize, width);
                  buildOptions += string(buf);
                }

            	size = 25;
                ifstream file(kernelFile);
                string prog(istreambuf_iterator<char>(file),(istreambuf_iterator<char>()));
                cl::Program::Sources source( 1, make_pair(prog.c_str(),prog.length()+1));
                cl::Program program(context, source);
                file.close();

		try{
			cerr << "buildOptions " << buildOptions << "\nQuantidade de Hash: " << (width * height)/25 <<endl;
            program.build(devices, buildOptions.c_str() );

		}
		catch(cl::Error& err){
                    cerr << "Build Failed! " << err.what()
                            << '(' << err.err() << ')' << endl;
                    cerr << "retrieving log ... " << endl;

                    exit(-1);

		}

                string kernelName = string(kernelFile).substr(0, string(kernelFile).find(".cl"));
                cerr << "especified kernel: " << kernelName << endl;
                kernel = cl::Kernel(program, kernelName.c_str());

                h_out = (uint64_t *)malloc(width*height*sizeof(uint64_t));
                h_data = (uint64_t *)malloc(width*height*sizeof(uint64_t));
                //h_cout = (uint64_t *)malloc(25*sizeof(uint64_t));
                //h_dout = (uint64_t *)malloc(25*sizeof(uint64_t));

                for(int i=0; i < width*height; i++){
                    h_data[i] = 0;
                    //if(i<25){
                    //h_cout[i] = 0;
                    //h_dout[i] = 0;}
                }

                d_data = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(uint64_t)*width*height);
                d_out = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(uint64_t)*width*height);
                //d_cout = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(uint64_t)*size);
                //d_dout = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(uint64_t)*size);

                kernel.setArg(0, d_data);
                kernel.setArg(1, d_out);
               // kernel.setArg(2, d_cout);
                //kernel.setArg(3, d_dout);
	}

        inline void initData(cl::CommandQueue& queue, cl::Event& event, int width, int height){
            queue.enqueueWriteBuffer(d_data, CL_TRUE, 0, sizeof(uint64_t)*width*height, h_data);
            //queue.enqueueWriteBuffer(d_cout, CL_TRUE, 0, sizeof(uint64_t)*size, h_cout);
            //queue.enqueueWriteBuffer(d_dout, CL_TRUE, 0, sizeof(uint64_t)*size, h_dout);
        }

        inline cl::Kernel& getKernel(){
            return kernel;
        }

        inline int goldenTest(cl::CommandQueue& queue, cl::Event& event, int width, int height){
            event.wait();
#ifdef PROFILING
            cl_ulong start = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
            cl_ulong end = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();

            double time = 1.e-9*(end-start);
            cout << "Time for kernel execute " << time << endl;
#endif
            queue.enqueueReadBuffer(d_out, CL_TRUE, 0, sizeof(uint64_t)*width*height, h_out);
            //queue.enqueueReadBuffer(d_cout, CL_TRUE, 0, sizeof(uint64_t)*size, h_cout);
            //queue.enqueueReadBuffer(d_dout, CL_TRUE, 0, sizeof(uint64_t)*size, h_dout);
            return(0);
        }

        cl::NDRange getGlobalWorkItems(int width, int height){
            return (cl::NDRange(width*5,height*5));
        }

        cl::NDRange getWorkItemsInWorkGroup(){
            return (cl::NDRange(5,5));
        }

        cl::NDRange getStartRange(){
            return (cl::NDRange(0,0));
        }

};

void displayinfo(cl::vector<cl::Platform> platformList, int deviceType){
    cout << "Platform Number is " << platformList.size() << endl;

    string platformVendor;
    platformList[0].getInfo((cl_platform_info)CL_PLATFORM_VENDOR, &platformVendor);
    cout << "Device Type: "
            <<((deviceType == CL_DEVICE_TYPE_GPU)?"GPU":"CPU") << endl;
    cout << "Platform by: " << platformVendor << "\n" << endl;
}

int main(int argc, char** argv) {

    if(argc < 1 ){
        cerr << "./keccakOpenCL cpu|gpu <kernel_file>" << endl;
        exit(EXIT_FAILURE);
    }

    int qtdHash = atoi(argv[2]);

    const string platformName("gpu");
    int deviceType = CL_DEVICE_TYPE_GPU;
    const char* kernelFile = argv[1];

    try{
        cl::vector<cl::Platform> platformList;
        cl::Platform::get(&platformList);

        displayinfo(platformList, deviceType);

        cl_context_properties cprops[3] = {
            CL_CONTEXT_PLATFORM,
            (cl_context_properties)(platformList[0]()),
            0
        };

        cl::Context context(deviceType, cprops);

        cl::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
#ifdef PROFILING
        cl::CommandQueue queue(context, devices[0], CL_QUEUE_PROFILING_ENABLE);
#else
        cl::CommandQueue queue(context, devices[0], 0);
#endif
        keccakOCL keccak(context, devices, kernelFile, qtdHash);

        cl::Event event;
        keccak.initData(queue, event, qtdHash*5, qtdHash*5);

        queue.enqueueNDRangeKernel(keccak.getKernel(),
                                 keccak.getStartRange(),
                                 keccak.getGlobalWorkItems(qtdHash, qtdHash),
                                 keccak.getWorkItemsInWorkGroup(),
                                 NULL,
                                 &event);
        if(keccak.goldenTest(queue, event, qtdHash*5, qtdHash*5) == 0){
            cout << "Test passed " << endl;
        }
        else{
            cout << "Test failed " << endl;
        }
    }catch(cl::Error error){
        cerr << "caught exception " << error.what()
                << '(' << error.err() << ')' << endl;
    }

    /*printf("\nOutput of GPU: [A]\n");
    for(int i=0;i<qtdHash*5*qtdHash*5;i++) {

        printf("%016lX ", h_out[i]);
	if((i+1)%5 == 0) printf("\n");
	if((i+1)%25 == 0) printf("\n\n");

    }printf("\n");*/


    return 0;
}
