CXX        = g++
LIB        = -L$(OPENCL_LIBDIR) -lOpenCL
CXXFLAGS   = -fopenmp -O3 -DWITH_FLOATS=0
NVCCFLAGS  = -O3 -DWITH_FLOATS=0

INCLUDES    += -I ../include 
GPU_OPTS   = -D lgWARP=5

SOURCES_CPP =ProjectMain.cpp ProjHelperFun.cpp 
HELPERS     =ProjHelperFun.h ../include/Constants.h ../include/ParseInput.h ../include/ParserC.h ../include/OpenmpUtil.h
OBJECTS     =ProjectMain.o ProjHelperFun.o  ProjCoreOrig.o
EXECUTABLE  =runproject


default: gpu

.cpp.o: $(SOURCES_CPP) $(HELPERS)
	$(CXX) $(CXXFLAGS) $(GPU_OPTS) $(INCLUDES) -c -o $@ $<

ProjCoreOrig.o: ProjCoreOrig.cu $(HELPERS) ../include/CudaUtilProj.cu.h
	nvcc $(NVCCFLAGS) $(GPU_OPTS) $(INCLUDES) -c -o $@ $<


gpu: $(EXECUTABLE)
$(EXECUTABLE): $(OBJECTS)
	nvcc $(NVCCFLAGS) $(INCLUDES) ProjHost.cu -o $(EXECUTABLE) $(OBJECTS)
	# nvcc -I../include/ ProjCoreOrig.cu ProjectMain.cpp ProjHost.cu  -o cudaexec

run_small: $(EXECUTABLE)
	cat ../Data/Small/input.data ../Data/Small/output.data | ./$(EXECUTABLE) 2> Debug.txt

run_medium: $(EXECUTABLE)
	cat ../Data/Medium/input.data ../Data/Medium/output.data | ./$(EXECUTABLE) 2> Debug.txt

run_large: $(EXECUTABLE)
	cat ../Data/Large/input.data ../Data/Large/output.data | ./$(EXECUTABLE) 2> Debug.txt

clean:
	rm -f Debug.txt $(EXECUTABLE) $(OBJECTS)

