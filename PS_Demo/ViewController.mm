//
//  ViewController.m
//  PS_Demo
//
//  Created by apple on 2017/4/21.
//  Copyright © 2017年 apple. All rights reserved.
//
#import "opencv2/core.hpp"
#import <opencv2/opencv.hpp>
#import <opencv2/imgproc/types_c.h>
#import <opencv2/imgcodecs/ios.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace cv;
using namespace std;  
#import "ViewController.h"
#import <ZXingObjC/ZXingObjC.h>
@interface ViewController (){
    cv::Mat cvImage;
}
@end



cv::Rect DrawXYProjection(const Mat image,Mat &imageOut,const int threshodValue,const int binaryzationValue) {
    Mat img=image.clone();
    if(img.channels()>1)
    {
        cvtColor(img,img,CV_RGB2GRAY);
    }
    Mat out(img.size(),img.type(),Scalar(255));
    imageOut=out;
    //对每一个传入的图片做灰度归一化，以便使用同一套阈值参数
    normalize(img,img,0,255,NORM_MINMAX);
    std::vector<int> vectorVertical(img.cols,0);
    for(int i=0;i<img.cols;i++)
    {
        for(int j=0;j<img.rows;j++)
        {
            if(img.at<uchar>(j,i)<binaryzationValue)
            {
                vectorVertical[i]++;
            }
        }
    }
    //列值归一化
    int high=img.rows/6;
    normalize(vectorVertical,vectorVertical,0,high,NORM_MINMAX);
    for(int i=0;i<img.cols;i++)
    {
        for(int j=0;j<img.rows;j++)
        {
            if(vectorVertical[i]>threshodValue)
            {
                
                line(imageOut,cv::Point(i,img.rows),cv::Point(i,img.rows-vectorVertical[i]),Scalar(0));
            }
        }
    }
    //水平投影
    std::vector<int> vectorHorizontal(img.rows,0);
    for(int i=0;i<img.rows;i++)
    {
        for(int j=0;j<img.cols;j++)
        {
            if(img.at<uchar>(i,j)<binaryzationValue)
            {
                vectorHorizontal[i]++;
            }
        }
    }
    normalize(vectorHorizontal,vectorHorizontal,0,high,NORM_MINMAX);
    for(int i=0;i<img.rows;i++)
    {
        for(int j=0;j<img.cols;j++)
        {
            if(vectorHorizontal[i]>threshodValue)
            {
                line(imageOut,cv::Point(img.cols-vectorHorizontal[i],i),cv::Point(img.cols,i),Scalar(0));
            }
        }
    }
    //找到投影四个角点坐标
    std::vector<int>::iterator beginV=vectorVertical.begin();
    std::vector<int>::iterator beginH=vectorHorizontal.begin();
    std::vector<int>::iterator endV=vectorVertical.end()-1;
    std::vector<int>::iterator endH=vectorHorizontal.end()-1;
    int widthV=0;
    int widthH=0;
    int highV=0;
    int highH=0;
    while(*beginV<threshodValue)
    {
        beginV++;
        widthV++;
    }
    while(*endV<threshodValue)
    {
        endV--;
        widthH++;
    }
    while(*beginH<threshodValue)
    {
        beginH++;
        highV++;
    }
    while(*endH<threshodValue)
    {
        endH--;
        highH++;
    }
    //投影矩形
    cv::Rect rect(widthV,highV,img.cols-widthH-widthV,img.rows-highH-highV);
    return rect;
}

@implementation ViewController







cv::Point Center_cal(vector<vector<cv::Point> > contours,int i)
{
    int centerx=0,centery=0,n=(int)contours[i].size();
    centerx = (contours[i][n/2].x + contours[i][n*2/4].x + contours[i][3*n/4].x + contours[i][n-1].x)/4;
    centery = (contours[i][n/4].y + contours[i][n*2/4].y + contours[i][3*n/4].y + contours[i][n-1].y)/4;
    cv::Point point1=cv::Point(centerx,centery);
    return point1;
}


cv::Rect QR_rect(vector<vector<cv::Point> > contours){
    int minX = 0,maxX = 0,minY = 0,maxY = 0;
    for (int i=0; i<contours.size(); i++) {
        int n =(int)contours[i].size();
        int mix=contours[i][0].x;
        int max = contours[i][n/2].x;
        int miy=contours[i][0].y;
        int may = contours[i][n/2].y;
        if (i==0 || minX>mix) {
            minX=mix;
        }
        if (i==0 || minY>miy) {
            minY=miy;
        }
        if (i==0 || maxX<max) {
            maxX=max;
        }
        if (i==0 || maxY<may) {
            maxY=may;
        }
    }
    
    
    
    return cv::Rect(minX-3, minY-3, maxX-minX+6, maxY-minY+6);
}


- (void)viewDidLoad {
    [super viewDidLoad];
    self.view.backgroundColor = [UIColor greenColor];
    // Do any additional setup after loading the view, typically from a nib.
    UIImage *img = [UIImage imageNamed:@"95CB1D65-C50B-4D84-980E-5D3D328305C9.jpg"];
    UIImageView *imageView=[[UIImageView alloc]initWithImage:img];
    imageView.frame = CGRectMake(10, 50, 400, 300);
    //imageView.contentMode = UIViewContentModeScaleAspectFit;
    [self.view addSubview:imageView];
    
    
    
    
    Mat image , imageGray, imageGuussian,imageSource;
    
    Mat imageSobelX,imageSobelY,imageSobelOut;
    
    UIImageToMat(img,imageSource);
    //resize(imageSource,imageSource,cv::Size(800,600));//标准大小
    
     Mat src_all=imageSource.clone();
    
    Mat threshold_output;
    
    vector<vector<cv::Point> > contours,contours2;
    vector<Vec4i> hierarchy;
    
    imageSource.copyTo(image);
    cvtColor(image, imageGray, CV_RGB2GRAY);
    blur( imageGray, imageGray, cv::Size(3,3) ); //模糊，去除毛刺
    
    //    Mat dst;
    //
    //      threshold( imageGray, dst, 100, 222,0 );
     threshold( imageGray, threshold_output, 100, 255, THRESH_BINARY );
    
    //寻找轮廓
    //第一个参数是输入图像 2值化的
    //第二个参数是内存存储器，FindContours找到的轮廓放到内存里面。
    //第三个参数是层级，**[Next, Previous, First_Child, Parent]** 的vector
    //第四个参数是类型，采用树结构
    //第五个参数是节点拟合模式，这里是全部寻找
    findContours( threshold_output, contours, hierarchy,  CV_RETR_TREE, CHAIN_APPROX_NONE, cv::Point(0, 0) );
    //轮廓筛选
    int c=0,ic=0,area=0;
    int parentIdx=-1;
    for( int i = 0; i< contours.size(); i++ )
    {
        //hierarchy[i][2] != -1 表示不是最外面的轮廓
        if (hierarchy[i][2] != -1 && ic==0)
        {
            parentIdx = i;
            ic++;
        }
        else if (hierarchy[i][2] != -1)
        {
            ic++;
        }
        //最外面的清0
        else if(hierarchy[i][2] == -1)
        {
            ic = 0;
            parentIdx = -1;
        }
        //找到定位点信息
        if ( ic >= 2)
        {
            contours2.push_back(contours[parentIdx]);
            ic = 0;
            parentIdx = -1;
        }
    }
    
    
    //填充定位点
  //  for(int i=0; i<contours2.size(); i++){
        
   // }
       //drawContours( src_all, contours2, i,  CV_RGB(0,255,0) , -1 );
    //连接定位点
//    cv::Point point[4];
//    for(int i=0; i<contours2.size(); i++)
//    {
//        point[i] = Center_cal( contours2, i );
//        
//        NSLog(@"************ %d          %d",point[i].x,point[i].y);
//        
//        
//        
//        
//    }
//    
//    line(src_all,point[0],point[1],Scalar(0,0,255),2);
//   line(src_all,point[1],point[2],Scalar(0,0,255),2);
//    line(src_all,point[2],point[3],Scalar(0,0,255),2);
//     line(src_all,point[0],point[3],Scalar(0,0,255),2);
    
    
    CvRect rect =QR_rect(contours2);
    
    IplImage imgTmp = _IplImage(imageSource);
    IplImage *input = cvCloneImage(&imgTmp);

    cvSetImageROI(input, rect);
    CvSize size = cvSize(rect.width, rect.height);
    
    IplImage* pDest = cvCreateImage(size,input->depth,input->nChannels);
    cvCopy(input,pDest); //复制图像
    cvResetImageROI(pDest);//源图像用完后，清空ROI

//    CvMat temp;
//    CvMat* mat = cvGetMat(pDest, &temp);
//    
// 
//    ;
    
    Mat im=cvarrToMat(pDest);
    
   //  src_all (pDest,true);
    
    
    UIImage *result = MatToUIImage(im);
    result = [self ScaleMini:result];
    UIImageView *simageView=[[UIImageView alloc]initWithImage:result ];
    simageView.frame = CGRectMake(10, 400, 400, 300);
    simageView.contentMode = UIViewContentModeScaleAspectFit;
    [self.view addSubview:simageView];
    
    CIDetector*detector = [CIDetector detectorOfType:CIDetectorTypeQRCode context:nil options:@{ CIDetectorAccuracy : CIDetectorAccuracyHigh }];
    NSArray *features = [detector featuresInImage:[CIImage imageWithCGImage:result.CGImage]];
    BOOL canOR = features.count >=1;
    if (canOR ){
        CIQRCodeFeature *feature = [features objectAtIndex:0];
        NSString *scannedResult = feature.messageString;
        NSLog(@"**********%@",scannedResult);
    }
    
    
    
    
   // resize(image, image, cv::Size(500,300));
    
    //转化为灰度图
   /* cvtColor(image, imageGray, CV_RGB2GRAY);
    
    //高斯平滑滤镜
    GaussianBlur(imageGray, imageGuussian, cv::Size(3,3), 0);
    
    //求得水平和垂直方向灰度图像的梯度差，使用Sobel算子
    Mat imageX16S,imageY16S;
    Sobel(imageGuussian, imageX16S, CV_16S, 1, 0,3,1,0,4);
     Sobel(imageGuussian, imageY16S, CV_16S, 0, 1,3,1,0,4);
    convertScaleAbs(imageX16S, imageSobelX,1,0);
    convertScaleAbs(imageY16S, imageSobelY,1,0);
    
    imageSobelOut = imageSobelX-imageSobelY;
    
    //5.均值滤波，消除高频噪声
    blur(imageSobelOut,imageSobelOut,cv::Size(3,3));
    
    //6.二值化
    Mat imageSobleOutThreshold;
    threshold(imageSobelOut,imageSobleOutThreshold,180,255,CV_THRESH_BINARY);
    
    //7.闭运算，填充条形码间隙
    Mat  element=getStructuringElement(0,cv::Size(7,7));
    morphologyEx(imageSobleOutThreshold,imageSobleOutThreshold,MORPH_CLOSE,element);
    
    //8. 腐蚀，去除孤立的点
    erode(imageSobleOutThreshold,imageSobleOutThreshold,element);
    
    //9. 膨胀，填充条形码间空隙，根据核的大小，有可能需要2~3次膨胀操作
    dilate(imageSobleOutThreshold,imageSobleOutThreshold,element);
    dilate(imageSobleOutThreshold,imageSobleOutThreshold,element);
    dilate(imageSobleOutThreshold,imageSobleOutThreshold,element);
    
    vector<vector<cv::Point>> contours;
    vector<Vec4i> hiera;
    
    
    //10.通过findContours找到条形码区域的矩形边界
    findContours(imageSobleOutThreshold,contours,hiera,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE);
    for(int i=0;i<contours.size();i++)
    {
        cv::Rect rect=boundingRect((Mat)contours[i]);
        rectangle(image,rect,Scalar(255),2);
    }*/
    
    
 /*   UIImageToMat(image,cvImage);
    
    cv:: Mat imageCopy = cvImage.clone();
    
    cv::Mat imageGray,imageOut;
    
    cvtColor(cvImage, imageGray, CV_RGB2GRAY);
    
   // cvCvtColor(&cvImage, &imageGray, CV_RGB2GRAY);
    cv::Rect rect(0,0,0,0);
    
    rect=   DrawXYProjection(cvImage,imageOut,cvImage.rows/10,100);
    
    cv::Mat roi = cvImage(rect);
    
    rectangle(imageCopy, cv::Point(rect.x,rect.y),cv::Point(rect.x+rect.width,rect.y+rect.height),  Scalar(100,0,255),2);
  //  cvRectangle(&imageCopy, cv::Point(rect.x,rect.y), cv::Point(rect.x+rect.width,rect.y+rect.height), cvScalar(0,0,255));*/
    
    
//    Mat dst;
//    
//      threshold( imageGray, dst, 100, 222,0 );
//    
//    UIImageToMat(img, cvImage);
//    cv::Mat gray;
//    // 将图像转换为灰度显示
//    cv::cvtColor(cvImage,gray,CV_RGB2GRAY);
//    // 应用高斯滤波器去除小的边缘
//    cv::GaussianBlur(gray, gray, cv::Size(5,5), 1.2,1.2);
//    // 计算与画布边缘
//    cv::Mat edges;
//    cv::Canny(gray, edges, 0, 50);
//    // 使用白色填充
//    cvImage.setTo(cv::Scalar::all(225));
//    // 修改边缘颜色
//    cvImage.setTo(cv::Scalar(0,128,255,255),edges);
//    
    
    
//    //高斯平滑滤镜
//    GaussianBlur(image, image, cv::Size(3,3), 0);
//     threshold(image,image,100,255,CV_THRESH_BINARY);  //二值化
//    Mat element=getStructuringElement(2,cv::Size(7,7));  //膨胀腐蚀核
//    
//    for(int i=0;i<10;i++)
//    {
//        erode(image,image,element);
//        i++;
//    }
//    Mat image1;
//    erode(image,image1,element);
//    image1=image-image1;
//    //寻找直线 边界定位也可以用findContours实现
//    vector<Vec2f>lines;
//    HoughLines(image1,lines,1,CV_PI/150,250,0,0);
//    Mat DrawLine=Mat::zeros(image1.size(),CV_8UC1);
//    for(int i=0;i<lines.size();i++)
//    {
//        float rho=lines[i][0];
//        float theta=lines[i][1];
//        cv:: Point pt1,pt2;
//        double a=cos(theta),b=sin(theta);
//        double x0=a*rho,y0=b*rho;
//        pt1.x=cvRound(x0+1000*(-b));
//        pt1.y=cvRound(y0+1000*a);
//        pt2.x=cvRound(x0-1000*(-b));
//        pt2.y=cvRound(y0-1000*a);
//        line(DrawLine,pt1,pt2,Scalar(255),1,CV_AA);
//    }
//    Point2f P1[4];
//    Point2f P2[4];
//    vector<Point2f>corners;
//    goodFeaturesToTrack(DrawLine,corners,4,0.1,10,Mat()); //角点检测
//    for(int i=0;i<corners.size();i++)
//    {
//        circle(DrawLine,corners[i],3,Scalar(255),3);
//        P1[i]=corners[i];
//    }
//    
//    int width=P1[1].x-P1[0].x;
//    int hight=P1[2].y-P1[0].y;
//    P2[0]=P1[0];
//    P2[1]=Point2f(P2[0].x+width,P2[0].y);
//    P2[2]=Point2f(P2[0].x,P2[1].y+hight);
//    P2[3]=Point2f(P2[1].x,P2[2].y);
//    Mat elementTransf;
//    elementTransf=  getAffineTransform(P1,P2);
//    warpAffine(imageSource,imageSource,elementTransf,imageSource.size(),1,0,Scalar(255));
//    
    
    
    



    
}
-(UIImage *)ScaleMini:(UIImage *)image{
    UIImage* bigImage = image;
    float actualHeight = bigImage.size.height;
    float actualWidth = bigImage.size.width;
    float newWidth =0;
    float newHeight =0;
    if(actualWidth > actualHeight) {
        //宽图
        newHeight =300.0f;
        newWidth = actualWidth / actualHeight * newHeight;
    }
    else
    {
        //长图
        newWidth =300.0f;
        newHeight = actualHeight / actualWidth * newWidth;
    }
    CGRect rect =CGRectMake(0.0,0.0, newWidth, newHeight);
    UIGraphicsBeginImageContext(rect.size);
    [bigImage drawInRect:rect];// scales image to rect
    image =UIGraphicsGetImageFromCurrentImageContext();
    UIGraphicsEndImageContext();
    //RETURN
    return image;
}
- (void)didReceiveMemoryWarning {
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
}


@end
