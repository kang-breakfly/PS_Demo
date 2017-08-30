//
//  ImageUtils.h
//  PS_Demo
//
//  Created by apple on 2017/4/21.
//  Copyright © 2017年 apple. All rights reserved.
//

#import <Foundation/Foundation.h>
#import <UIKit/UIKit.h>
@interface ImageUtils : NSObject
/**
 *美白效果
 */
+(UIImage *)imageWhitening:(UIImage *)image;
/**
 *openVC 处理图片的问题
 */
+(UIImage *)openVCImageWhitening:(UIImage *)image;
@end
