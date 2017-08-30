//
//  ImageUtils.m
//  PS_Demo
//
//  Created by apple on 2017/4/21.
//  Copyright © 2017年 apple. All rights reserved.
//

#import "ImageUtils.h"
#import "ColorUtils.h"
@implementation ImageUtils
+(UIImage *)imageWhitening:(UIImage *)image{
    //第一步获取图片的大小
    //第一种  size    image.size.width   image.size.height
    //第二种CGImage 获取
    CGImageRef imageRef = [image CGImage];
    NSUInteger width = CGImageGetWidth(imageRef);
    NSUInteger heigth = CGImageGetHeight(imageRef);
    //第二部：创建颜色空间（灰色空间、彩色空间）-> 开辟一块内存空间
    CGColorSpaceRef colorSpaceRef = CGColorSpaceCreateDeviceRGB();
 
    //第三步：创建图片上下文（解析图片信息）
    UInt32 *inputPixels = (UInt32*)calloc(width*heigth,sizeof(UInt32));
    
    CGContextRef contextRef = CGBitmapContextCreate(inputPixels, width, heigth, 8, width*4, colorSpaceRef, kCGImageAlphaPremultipliedFirst|kCGBitmapByteOrder32Big);
    
    CGContextDrawImage(contextRef, CGRectMake(0, 0, width, heigth), imageRef);
    
    //处理像素
    int lumi = 50;
    for (int i=0; i<heigth; i++) {
        for (int j=0; j<width; j++) {
            UInt32 * currentPixels = inputPixels + (i*width)+j;
            UInt32 color = *currentPixels;
            UInt32 thisR,thisG,thisB,thisA;
            
            thisR =R(color);
            thisR = thisR+lumi>255?255:thisR+lumi;
            
            thisG = G(color);
            thisG = thisG+lumi>255?255:thisG+lumi;
            
            thisB = B(color);
            thisB = thisB+lumi>255?255:thisB+lumi;
            
            thisA = A(color);
            
            *currentPixels = RGBAMake(thisR, thisG, thisB, thisA);
        }
    }
    //创建新图
    CGImageRef newImageRef = CGBitmapContextCreateImage(contextRef);
    UIImage *newImage=[UIImage imageWithCGImage:newImageRef];
    //释放内存
    CGColorSpaceRelease(colorSpaceRef);
    CGContextRelease(contextRef);
    CGImageRelease(newImageRef);
    
    free(inputPixels);
    
    return newImage;
}
/**
 *openVC 处理图片的问题
 */
+(UIImage *)openVCImageWhitening:(UIImage *)image{
    
    return nil;
}
@end
