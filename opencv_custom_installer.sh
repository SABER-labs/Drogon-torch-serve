mkdir -p opencv_install_directory
cd opencv_install_directory

[ ! -d "opencv/" ] && git clone --depth 1 --branch 4.5.5 https://github.com/opencv/opencv
[ ! -d "opencv_contrib/" ] && git clone --depth 1 --branch 4.5.5 https://github.com/opencv/opencv_contrib

if [ ! -d "libjpeg-turbo-2.1.2/" ]; then
  wget https://github.com/libjpeg-turbo/libjpeg-turbo/archive/refs/tags/2.1.2.zip && unzip 2.1.2.zip
  mkdir -p libjpeg-turbo-2.1.2/build
  cd libjpeg-turbo-2.1.2/build
  CFLAGS="-fPIC" cmake -G"Unix Makefiles" ..
  make -j$(nproc --all) && sudo make install
  cd ../..
  rm -rf 2.1.2.zip
fi

sudo apt update && sudo apt install -y --no-install-recommends \
	build-essential cmake pkg-config yasm checkinstall autoconf automake \
	libtool ca-certificates libjpeg-dev libpng-dev libtiff-dev libavcodec-dev \
	libavformat-dev libswscale-dev libgstreamer1.0-dev \
	libgstreamer-plugins-base1.0-dev libxvidcore-dev x264 libx264-dev \
	libfaac-dev libmp3lame-dev libtheora-dev libfaac-dev libmp3lame-dev libvorbis-dev \
	libopencore-amrnb-dev libopencore-amrwb-dev libdc1394-22-dev libxine2-dev \
	libv4l-dev v4l-utils libprotobuf-dev protobuf-compiler swig libtbb-dev libatlas-base-dev \
	gfortran libgoogle-glog-dev libgflags-dev libgphoto2-dev libeigen3-dev libhdf5-dev doxygen \
	libtesseract-dev liblapacke-dev zip unzip

if [ ! -d "opencv/build/" ]; then
  mkdir -p opencv/build && cd opencv/build
  cmake -GNinja -D CMAKE_BUILD_TYPE=RELEASE \
  -D CMAKE_INSTALL_PREFIX=/usr/local \
  -D WITH_JPEG=ON \
  -D BUILD_JPEG=OFF \
  -D JPEG_INCLUDE_DIR=/opt/libjpeg-turbo/include/ \
  -D JPEG_LIBRARY=/opt/libjpeg-turbo/lib64/libjpeg.a \
  -D WITH_TBB=ON \
  -D ENABLE_FAST_MATH=1 \
  -D WITH_V4L=OFF \
  -D WITH_QT=OFF \
  -D WITH_OPENGL=ON \
  -D WITH_GSTREAMER=ON \
  -D OPENCV_GENERATE_PKGCONFIG=ON \
  -D OPENCV_PC_FILE_NAME=opencv.pc \
  -D OPENCV_ENABLE_NONFREE=ON \
  -D BUILD_opencv_python3=YES \
  -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
  -D INSTALL_PYTHON_EXAMPLES=OFF \
  -D INSTALL_C_EXAMPLES=OFF \
  -D BUILD_SHARED_LIBS=ON \
  -D BUILD_EXAMPLES=OFF ..
  ninja
  sudo ninja install
  cd ../..
fi

sudo /bin/bash -c 'echo "/usr/local/lib" >> /etc/ld.so.conf.d/opencv.conf'
sudo ldconfig