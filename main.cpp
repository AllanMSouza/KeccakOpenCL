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
	keccakOCL(cl::Context& context, cl::vector<cl::Device>& devices, const char* kernelFile){

                size = 25;
                ifstream file(kernelFile);
                string prog(istreambuf_iterator<char>(file),(istreambuf_iterator<char>()));
                cl::Program::Sources source( 1, make_pair(prog.c_str(),prog.length()+1));
                cl::Program program(context, source);
                file.close();


		try{
			program.build(devices, NULL);
                        cout << "hahaha" << endl;

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

                h_out = (uint64_t *)malloc(25*sizeof(uint64_t));
                h_data = (uint64_t *)malloc(25*sizeof(uint64_t));
                h_cout = (uint64_t *)malloc(25*sizeof(uint64_t));
                h_dout = (uint64_t *)malloc(25*sizeof(uint64_t));

                for(int i=0; i < 25; i++){
                    h_data[i] = 0;
                    h_cout[i] = 0;
                    h_dout[i] = 0;
                }

                d_data = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(uint64_t)*size);
                d_out = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(uint64_t)*size);
                d_cout = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(uint64_t)*size);
                d_dout = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(uint64_t)*size);

                kernel.setArg(0, d_data);
                kernel.setArg(1, d_out);
                kernel.setArg(2, d_cout);
                kernel.setArg(3, d_dout);
	}

        inline void initData(cl::CommandQueue& queue, cl::Event& event){
            queue.enqueueWriteBuffer(d_data, CL_TRUE, 0, sizeof(uint64_t)*size, h_data);
            queue.enqueueWriteBuffer(d_cout, CL_TRUE, 0, sizeof(uint64_t)*size, h_cout);
            queue.enqueueWriteBuffer(d_dout, CL_TRUE, 0, sizeof(uint64_t)*size, h_dout);
        }

        inline cl::Kernel& getKernel(){
            return kernel;
        }

        inline int goldenTest(cl::CommandQueue& queue, cl::Event& event){
            event.wait();
#ifdef PROFILING
            cl_ulong start = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
            cl_ulong end = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();

            double time = 1.e-9*(end-start);
            cout << "Time for kernel execute " << time << endl;
#endif
            queue.enqueueReadBuffer(d_out, CL_TRUE, 0, sizeof(uint64_t)*size, h_out);
            queue.enqueueReadBuffer(d_cout, CL_TRUE, 0, sizeof(uint64_t)*size, h_cout);
            queue.enqueueReadBuffer(d_dout, CL_TRUE, 0, sizeof(uint64_t)*size, h_dout);
            return(0);
        }

        cl::NDRange getGlobalWorkItems(){
            return (cl::NDRange(25));
        }

        cl::NDRange getWorkItemsInWorkGroup(){
            return (cl::NDRange(25,0));
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

int teste(int argc, char** argv) {

    if(argc == 0){
        cerr << "./keccakOpenCL <kernel_file>" << endl;
        exit(EXIT_FAILURE);
    }

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
        keccakOCL keccak(context, devices, kernelFile);

        cl::Event event;
        keccak.initData(queue, event);

        queue.enqueueNDRangeKernel(keccak.getKernel(),
                                 keccak.getStartRange(),
                                 keccak.getGlobalWorkItems(),
                                 keccak.getWorkItemsInWorkGroup(),
                                 NULL,
                                 &event);
        if(keccak.goldenTest(queue, event) == 0){
            cout << "Test passed " << endl;
        }
        else{
            cout << "Test failed " << endl;
        }
    }catch(cl::Error error){
        cerr << "caught exception " << error.what()
                << '(' << error.err() << ')' << endl;
    }

    printf("\nOutput of GPU: [A]\n");
    for(int i=0;i<25;++i) {
        if(i%5 == 0) printf("\n");
        printf("%016lX ", h_out[i]);
    }printf("\n");

    printf("\nOutput of GPU: [C]\n");
    for(int i=0;i<25;++i) {
        if(i%5 == 0) printf("\n");
        printf("%016lX ", h_cout[i]);
    }printf("\n");

    printf("\nOutput of GPU: [D]\n");
    for(int i=0;i<25;++i) {
        if(i%5 == 0) printf("\n");
        printf("%016lX ", h_dout[i]);
    }printf("\n");

    return 0;
}

