INCLUDE_DIRS = -I/usr/include/opencv4
LIB_DIRS = 
CXX = g++

CDEFS=
CXXFLAGS= -O0 -g $(INCLUDE_DIRS) $(CDEFS)
OPENCV_LIBS= -L/usr/lib -lopencv_core -lopencv_flann -lopencv_video -lrt 

LIBS=-lpthread

PRODUCT=LaneDetect

HFILES= LaneDetect.h
CFILES= LaneDetect.cpp main.cpp

SRCS= ${HFILES} ${CFILES}
OBJS= $(CFILES:.cpp=.o)

all: ${PRODUCT}

$(PRODUCT): $(OBJS)
	$(CXX) $(LDFLAGS) $(CXXFLAGS) -o $@ $(OBJS) `pkg-config --libs opencv4` $(OPENCV_LIBS)

clean:
	-rm -f *.o *.NEW *~
	-rm -f ${PRODUCT} ${DERIVED} ${GARBAGE}

depend:

.cpp.o:
	$(CXX) $(CXXFLAGS) -c $<