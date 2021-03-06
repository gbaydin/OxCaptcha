/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package CaptchaTest;

import java.awt.Color;
import robots.OxCaptcha.*;
import java.awt.image.BufferedImage;
import java.io.*;
import java.util.Arrays;

/**
 *
 * @author gunes
 */
public class Test {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        // TODO code application logic here
        
        // Create Captcha container
        OxCaptcha c = new OxCaptcha(200, 50);

        long startTime = System.currentTimeMillis();

        // Create background
        c.background();
        
        c.setFont("DejaVu Serif", 1, 35);
//        c.setHollow();
        
        // Add text
//        c.text(8);
//        c.text("6AAK");
        c.textCentered("bacitly", 1, -2);

//          c.text("trustother", 3, 40, 3);
//        c.text(new char[] {'4', 'c', 'z', '8', 'J', 'y', 'A', 'z'}, new int[] {1,1,1,1,1,1,1,1,1}, 0, 30, -4);

//        c.text(new char[] {'a', 'b', 'c'}, 0, 30, -2);
//        c.text("4cz8JyAz", 0);
//          c.text(new char[] {'a', 'b', 'c', 'd'}, new int[] {0, 1, 2, 3}, 0, 30, -2);
//          c.textCentered(new char[] {'a', 'b', 'c', 'd'}, new int[] {0, 1, 2, 3}, -2);
//          c.textCentered(new char[] {'a', 'b', 'c', 'd'}, new int[] {0, 1, 2, 3}, -2);
        
        // Apply distortion
        //c.distortionFishEye();
        //c.distortionStretch(1.4, 1.1);
//        c.distortionShear(10, 10, 10, 10);
//        c.distortionShear();
//        c.distortionShear(35, 2, 35, 10);
//        c.distortionElectric();
          c.distortionElastic();
        c.distortionShear2();
//          c.distortionShear2(200, 15, 10, 15, 15, 10);

        c.recenter();
        // Add noise
//        c.noiseStraightLine();
//        c.noiseCurvedLine();
//        c.noiseCurvedLine(15, 100, 2.0f);
//        c.noiseSaltPepper();
        //c.noiseSaltPepper(0.05f, 0.05f);

        // Add blur
        //c.blur(10);
        c.blurGaussian(0.7);
        //c.blurGaussian5x5s1();
        //c.blurGaussian5x5s2();

//        c.normalize();
       
        // Get rendered image
        //BufferedImage img = c.getImage();
        
        // Get rendered image as a 2D array
        //int[] imgArray = c.getImageArray1D();
        
        long stopTime = System.currentTimeMillis();
        long elapsedTime = stopTime - startTime;

        //System.out.println(Arrays.toString(imgArray));
        
        // Write rendered image to a png file
        System.out.println(Long.toString(elapsedTime) + " ms");
        try {
            c.save("test.png");
        }
        catch (IOException e) {
            e.printStackTrace();
        }
        
    }
    
}
