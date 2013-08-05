/*
 * File:   main.cpp
 * Author: allan
 *
 * Created on February 26, 2013, 7:07 PM
 */

#define PROFILING // Define to see the time the kernel takes
#define __NO_STD_VECTOR // Use cl::vector instead of STL version
#define __CL_ENABLE_EXCEPTIONS // needed for exceptions

#define MAX_FILE_SIZE 5000000
#define BITRATE       1024
#define DATA_SIZE     384

#define PRINT_GPU_RESULT           \
    printf("\nOutput of GPU:\n");  \
    for(int i=0;i<25;++i) {       \
        printf("%016lX ", h_out[i]); \
        if((i+1)%5 == 0) \
            printf("\n"); \
    } printf("\n\n\n");

#include <cstdlib>
#include <CL/cl.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cmath>
#include <errno.h>
#include <sys/stat.h>
#include <inttypes.h>

using namespace std;

static uint64_t  *h_data;
static uint64_t  *h_out;

static uint64_t A[25] =     {0x0, 0x0, 0x0, 0x0, 0x0,
                             0x0, 0x0, 0x0, 0x0, 0x0,
                             0x0, 0x0, 0x0, 0x0, 0x0,
                             0x0, 0x0, 0x0, 0x0, 0x0,
                             0x0, 0x0, 0x0, 0x0, 0x0
                            };

class oclKeccak
{
private:

    cl::Kernel kernel;
    cl::Buffer d_data, d_out, d_digestlenght;
    cl_int size;

public:
    oclKeccak(cl::Context& context, cl::vector<cl::Device>& devices,
              const char* kernelFile)
    {

        size = 25;
        ifstream file(kernelFile);
        string prog(istreambuf_iterator<char>(file),(istreambuf_iterator<char>()));
        cl::Program::Sources source( 1, make_pair(prog.c_str(),prog.length()+1));
        cl::Program program(context, source);
        file.close();

        try
        {
            cerr << "buildOptions " << " " << endl;
            program.build(devices, NULL);
        }
        catch(cl::Error& err)
        {
            // Get the build log
            cerr << "Build failed! " << err.what()
                 << '(' << err.err() << ')' << endl;
            cerr << "retrieving  log ... " << endl;
            cerr
                    << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0])
                    << endl;
            exit(-1);
        }

        string kernelName = string(kernelFile).substr(0, string(kernelFile).find(".cl"));
        cerr << "especified kernel: " << kernelName << endl;
        kernel = cl::Kernel(program, kernelName.c_str());

        h_out = (uint64_t *)malloc(25*sizeof(uint64_t));

        d_data = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(uint64_t)*size);
        d_out  = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(uint64_t)*size);


        kernel.setArg(0, d_data);
        kernel.setArg(1, d_out);

    }

    inline void initData(cl::CommandQueue& queue, cl::Event& event)
    {
        queue.enqueueWriteBuffer(d_data, CL_TRUE, 0, sizeof(uint64_t)*size, A);
        //queue.enqueueWriteBuffer(d_out, CL_TRUE, 0, sizeof(uint64_t)*size, h_out);
    }

    inline cl::Kernel& getKernel()
    {
        return(kernel);
    }



    inline int goldenTest(cl::CommandQueue& queue, cl::Event& event)
    {
        event.wait();
#ifdef PROFILING
        cl_ulong start=
            event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
        cl_ulong end=
            event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
        double time = 1.e-9 * (end-start);
        cout << "Time for kernel to execute " << time << endl;
#endif
        queue.enqueueReadBuffer(d_out, CL_TRUE, 0, sizeof(uint64_t)*size, h_out);
        return(0);
    }

    cl::NDRange getGlobalWorkItems()
    {
        return( cl::NDRange(1) );
    }

    cl::NDRange getWorkItemsInWorkGroup()
    {
        return( cl::NDRange(1) );
    }

};

void displayInfo(cl::vector<cl::Platform> platformList, int deviceType)
{
    cout << "Platform Number is: " << platformList.size() << endl;

    string platformVendor;
    platformList[0].getInfo((cl_platform_info)CL_PLATFORM_VENDOR, &platformVendor);
    cout << "Device Type: "
         << ((deviceType == CL_DEVICE_TYPE_GPU)?"GPU":"CPU") << endl;
    cout << "Platform by: " << platformVendor << "\n" <<endl;
}

int main(int argc, char** argv)
{

    if( argc < 2)
    {
        cerr << "Use: ./Keccak kernelFile " << endl;
        exit(EXIT_FAILURE);
    }

    //int threads = atoi(argv[2]);

    const string platformName(argv[1]);
    int deviceType = CL_DEVICE_TYPE_GPU;
    const char* kernelFile = argv[1];

    try
    {
        cl::vector<cl::Platform> platformList;
        cl::Platform::get(&platformList);

        displayInfo(platformList, deviceType);

        cl_context_properties cprops[3] =
        {
            CL_CONTEXT_PLATFORM,
            (cl_context_properties)(platformList[0])(), 0
        };

        cl::Context context(deviceType, cprops);

        cl::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();

#ifdef PROFILING
        cl::CommandQueue queue(context, devices[0], CL_QUEUE_PROFILING_ENABLE);
#else
        cl::CommandQueue queue(context, devices[0], 0);
#endif

        oclKeccak keccak(context, devices, kernelFile);

        cl::Event event;
        keccak.initData(queue, event);

        queue.enqueueNDRangeKernel(keccak.getKernel(),
                                   cl::NullRange,
                                   keccak.getGlobalWorkItems(),
                                   keccak.getWorkItemsInWorkGroup(),
                                   NULL,
                                   &event);
        // PRINT_GPU_RESULT;

        if(keccak.goldenTest(queue, event) == 0)
        {
            cout << "test passed" << endl;
        }
        else
        {
            cout << "TEST FAILED!" << endl;
        }

    }
    catch (cl::Error error)
    {
        cerr << "caught exception: " << error.what()
             << '(' << error.err() << ')' << endl;
    }

    PRINT_GPU_RESULT;

    return 0;
}
