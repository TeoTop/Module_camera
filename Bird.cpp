#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <stdio.h>
#include <time.h>
#include <unistd.h>
#include <stdlib.h>
#include <cmath> 
#include <math.h> 
#include <map>
#include <iterator>
#include <list>
#include <algorithm>
using namespace cv;
using namespace std;
#define XSEC 1

//motionTracking.cpp

//Written by  Kyle Hounslow, December 2013

//Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software")
//, to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
//and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

//The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

//THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
//LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
//IN THE SOFTWARE.

//our sensitivity value to be used in the absdiff() function
const static int SENSITIVITY_VALUE = 20;
//size of blur used to smooth the intensity image output from absdiff() function
const static int BLUR_SIZE = 10;
//we'll have just one object to search for
//and keep track of its position.
int theObject[2] = {0,0};
int counter = 0;
time_t timer1 = 0;
time_t timer2 = 0;
double diftime;
bool test = false;
int cameraWidth;
int cameraHeight;
string rasp = "1";
//string path = "/mnt/upload/";
string path = "";
//bounding rectangle of the object, we will use the center of this as its position.
Rect objectBoundingRectangle = Rect(0,0,0,0);
//structure color
typedef struct{
	int R;int G;int B;
}Color;
typedef struct 
{
  int R;int G;int B;
}maxpic;
map<string, Color> allcolor;

//int to string helper function
string intToString(int number){

	//this function has a number input and string output
	std::stringstream ss;
	ss << number;
	return ss.str();
}

Color dominantColor(Mat src){
	Color c; 
	maxpic maxi = {0,0,0};// represent pic
	/// Separate the image in 3 places ( B, G and R )
	vector<Mat> bgr_planes;
	split( src, bgr_planes );

	/// Establish the number of bins
	int histSize = 256;

	/// Set the ranges ( for B,G,R) )
	float range[] = { 0, 256 } ;
	const float* histRange = { range };

	bool uniform = true; bool accumulate = false;
	Mat b_hist, g_hist, r_hist;

	/// Compute the histograms:
	calcHist( &bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate );
	calcHist( &bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate );
	calcHist( &bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate );

	// Draw the histograms for B, G and R
	int hist_w = 512; int hist_h = 400;
	int bin_w = cvRound( (double) hist_w/histSize );

	Mat histImage( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );

	/// Normalize the result to [ 0, histImage.rows ]
	normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
	normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
	normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
	for( int i = 1; i < histSize; i++ )
	{
		maxpic tmp = {max(maxi.R,cvRound(r_hist.at<float>(i))),max(maxi.G,cvRound(g_hist.at<float>(i))),max(maxi.B,cvRound(b_hist.at<float>(i)))};
		if (tmp.R > maxi.R) {
			maxi.R = tmp.R;
			c.R = i;
		}
		if (tmp.G > maxi.G) {
			maxi.G = tmp.G;
			c.G = i;
		}
		if (tmp.B > maxi.B) {
			maxi.B = tmp.B;
			c.B = i;
		}
	}
	return c;
}
string rgbtostring(Color c){
	int min = pow((c.R-allcolor.begin()->second.R),2) +
	 pow((c.G-allcolor.begin()->second.G),2) + 
	 pow((c.B-allcolor.begin()->second.B),2);
	string colorname;
	for(std::map<string, Color>::iterator it=allcolor.begin();it!=allcolor.end();it++){
		int a = pow((c.R-it->second.R),2) + pow((c.G-it->second.G),2) + pow((c.B-it->second.B),2);
		if(min >= a) {
			colorname = it->first;
			min = a;
		}
	}
	return colorname;
}
void searchForMovement(Mat thresholdImage, Mat &cameraFeed){
	//notice how we use the '&' operator for objectDetected and cameraFeed. This is because we wish
	//to take the values passed into the function and manipulate them, rather than just working with a copy.
	//eg. we draw to the cameraFeed to be displayed in the main() function.
	bool objectDetected = false;
	Mat temp;
	thresholdImage.copyTo(temp);
	//these two vectors needed for output of findContours
	vector< vector<Point> > contours;
	vector<Vec4i> hierarchy;
	//find contours of filtered image using openCV findContours function
	//findContours(temp,contours,hierarchy,CV_RETR_CCOMP,CV_CHAIN_APPROX_SIMPLE );// retrieves all contours
	findContours(temp,contours,hierarchy,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_SIMPLE );// retrieves external contours
	//if contours vector is not empty, we have found some object
	if(contours.size()>0){objectDetected=true;}
	else {objectDetected = false;}

	if(objectDetected){
		//the largest contour is found at the end of the contours vector
		//we will simply assume that the biggest contour is the object we are looking for.
		vector< vector<Point> > largestContourVec;
		largestContourVec.push_back(contours.at(contours.size()-1));
		//make a bounding rectangle around the largest contour then find its centroid
		//this will be the object's final estimated position.
		objectBoundingRectangle = boundingRect(largestContourVec.at(0));
		int xpos = objectBoundingRectangle.x+objectBoundingRectangle.width/2;
		int ypos = objectBoundingRectangle.y+objectBoundingRectangle.height/2;

		//update the objects positions by changing the 'theObject' array values
		theObject[0] = xpos , theObject[1] = ypos;

		//drawContours(cameraFeed,largestContourVec,-1,Scalar(0,0,255),2);
		
		time(&timer2);
  		diftime = difftime(timer2,timer1);
  		//si > 1/3 taille de la camera 
		  if(diftime >= XSEC && objectBoundingRectangle.width >= cameraWidth/4 && objectBoundingRectangle.height >= cameraHeight/4) {
		    time(&timer1);
		    
		    /*****/
		    //extract region which moved
		    vector<Mat> subregions;
	        Mat contourRegion;
		    for (int i = 0; i < largestContourVec.size(); i++)
		    {
		        // Get bounding box for contour
		        Rect roi = boundingRect(largestContourVec[i]); // This is a OpenCV function

		        // Create a mask for each contour to mask out that region from image.
		        Mat mask = Mat::zeros(cameraFeed.size(), CV_8UC1);
		        drawContours(mask, largestContourVec, i, Scalar(255), CV_FILLED); // This is a OpenCV function

		        // At this point, mask has value of 255 for pixels within the contour and value of 0 for those not in contour.

		        // Extract region using mask for region
		        Mat imageROI;
		        cameraFeed.copyTo(imageROI, mask); // 'image' is the image you used to compute the contours.
		        contourRegion = imageROI(roi);
		        // Mat maskROI = mask(roi); // Save this if you want a mask for pixels within the contour in contourRegion. 

		        // Store contourRegion. contourRegion is a rectangular image the size of the bounding rect for the contour 
		        // BUT only pixels within the contour is visible. All other pixels are set to (0,0,0).
		        subregions.push_back(contourRegion);
		    }
	       // imshow("contour",contourRegion);
		    /****/

		    vector<int> compression_params;
		    compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
		    compression_params.push_back(9);//9 higher compression
		    Mat smallImage = cv::Mat(cameraFeed, objectBoundingRectangle).clone();
		    //imshow("image",smallImage);
			cvtColor(contourRegion,contourRegion,CV_BGR2RGB);    
			Color c = dominantColor(contourRegion);
			//imshow("contour",contourRegion);
		    struct tm *local_time;
		    local_time = localtime(&timer1);
		    string id = intToString(local_time->tm_mday)+intToString(local_time->tm_hour)+intToString(local_time->tm_min)+intToString(local_time->tm_sec);
		    string im_name = "";
		    im_name += rasp; 
		    im_name += id;
		    int taille = (int)sqrt(((double)objectBoundingRectangle.width)*((double)objectBoundingRectangle.width) + ((double)objectBoundingRectangle.height)*((double)objectBoundingRectangle.height));
		    string t = "";
		    if(taille <= 1/3*cameraHeight){
		    	t += "petit";

		    }else if (taille <=2/3*cameraHeight){
		    	t+= "moyen";
		    }else{
		    	t+= "grand";
		    }
  		    string color = rgbtostring(c);
		    im_name += "@" + color + "@" + t + ".png";
		    cout << c.R << " " << c.G << " " << c.B << endl;

		  //cout << moyenneB << endl;
			cout << path + im_name << endl;
			Mat imagewrite;
			cvtColor(smallImage,imagewrite,CV_BGR2RGB);
		    imwrite(path + im_name, imagewrite, compression_params);
		    //imwrite(path + im_name, smallImage, compression_params);
		  }
	}
	//make some temp x and y variables so we dont have to type out so much
	//int x = theObject[0];
	//int y = theObject[1];
	
	//draw some crosshairs around the object

	//line(cameraFeed,Point(x,y),Point(x,y-25),Scalar(0,255,0),2);
	//line(cameraFeed,Point(x,y),Point(x,y+25),Scalar(0,255,0),2);
	//line(cameraFeed,Point(x,y),Point(x-25,y),Scalar(0,255,0),2);
	//line(cameraFeed,Point(x,y),Point(x+25,y),Scalar(0,255,0),2);

	//write the position of the object to the screen
	//putText(cameraFeed,"Tracking object at (" + intToString(x)+","+intToString(y)+")",Point(x,y),1,1,Scalar(255,0,0),2);

	

}
int main(){

	//some boolean variables for added functionality
	bool objectDetected = true;
	//these two can be toggled by pressing 'd' or 't'
	bool debugMode = false;
	bool trackingEnabled = true;
	//pause and resume code
	bool pause = false;
	//set up the matrices that we will need
	//the two frames we will be comparing
	Mat frame1,frame2;
	//their grayscale images (needed for absdiff() function)
	Mat grayImage1,grayImage2;
	//resulting difference image
	Mat differenceImage;
	//thresholded difference image (for use in findContours() function)
	Mat thresholdImage;
	//video capture object.
	VideoCapture capture(-1);

	/***Color***/
	allcolor["Bleu"] = {0, 0, 255};allcolor["Aigue-marine"] = {121, 248, 248};allcolor["Azur"] = {0, 127, 255};allcolor["Azur"] = {30, 127, 203};allcolor["Azur_clair"] = {116, 208, 241};allcolor["Azurin"] = {169, 234, 254};allcolor["Bleu_acier"] = {58, 142, 186};allcolor["Bleu_ardoise"] = {104, 111, 140};allcolor["Bleu_barbeau"] = {84, 114, 174};allcolor["Bleu_bleuet"] = {84, 114, 174};allcolor["Bleu_bondi"] = {0, 149, 182};allcolor["Bleu_céleste"] = {38, 196, 236};allcolor["Bleu_céruléen"] = {15, 157, 232};allcolor["Bleu_céruléen"] = {53, 122, 183};allcolor["Bleu_charrette"] = {142, 162, 198};allcolor["Bleu_charron"] = {23, 101, 125};allcolor["Bleu_charron"] = {142, 162, 198};allcolor["Bleu_ciel"] = {119, 181, 254};allcolor["Bleu_cobalt"] = {34, 66, 124};allcolor["Bleu_de_Berlin"] = {36, 68, 92};allcolor["Bleu_de_France"] = {49, 140, 231};allcolor["Bleu_de_minuit"] = {0, 51, 102};allcolor["Bleu_de_Prusse"] = {36, 68, 92};allcolor["Bleu_denim"] = {21, 96, 189};allcolor["Bleu_des_mers_du_sud"] = {0, 204, 203};allcolor["Bleu_dragée"] = {223, 242, 255};allcolor["Bleu_égyptien"] = {16, 52, 166};allcolor["Bleu_électrique"] = {44, 117, 255};allcolor["Bleu_guède"] = {86, 115, 154};allcolor["Bleu_horizon"] = {127, 143, 166};allcolor["Bleu_majorelle"] = {96, 80, 220};allcolor["Bleu_marine"] = {3, 34, 76};allcolor["Bleu_maya"] = {115, 194, 251};allcolor["Bleu_minéral"] = {36, 68, 92};allcolor["Bleu_nuit"] = {15, 5, 107};allcolor["Bleu_outremer"] = {27, 1, 155};allcolor["Bleu_outremer"] = {43, 0, 154};allcolor["Bleu_paon"] = {6, 119, 144};allcolor["Bleu_persan"] = {102, 0, 255};allcolor["Bleu_pétrole"] = {29, 72, 81};allcolor["Bleu_roi"] = {49, 140, 231};allcolor["Bleu_saphir"] = {1, 49, 180};allcolor["Bleu_sarcelle"] = {0, 142, 142};allcolor["Bleu_smalt"] = {0, 51, 153};allcolor["Bleu_tiffany"] = {10, 186, 181};allcolor["Bleu_turquin"] = {66, 91, 138};allcolor["Cæruléum"] = {38, 196, 236};allcolor["Canard"] = {4, 139, 154};allcolor["Cérulé"] = {116, 208, 241};allcolor["Cyan"] = {0, 255, 255};allcolor["Cyan"] = {43, 250, 250};allcolor["Fumée"] = {187, 210, 225};allcolor["Givré"] = {128, 208, 208};allcolor["Indigo"] = {121, 28, 248};allcolor["Indigo"] = {46, 0, 108};allcolor["Indigo_du_web"] = {75, 0, 130};allcolor["Klein"] = {0, 47, 167};allcolor["Klein"] = {33, 23, 125};allcolor["Lapis-lazuli"] = {38, 97, 156};allcolor["Lavande"] = {150, 131, 236};allcolor["Pastel"] = {86, 115, 154};allcolor["Pervenche"] = {204, 204, 255};allcolor["Turquoise"] = {37, 253, 233};allcolor["Blanc"] = {255, 255, 255};allcolor["Albâtre"] = {254, 254, 254};allcolor["Argile"] = {239, 239, 239};allcolor["Azur_brume"] = {240, 255, 255};allcolor["Beige_clair"] = {245, 245, 220};allcolor["Blanc_cassé"] = {254, 254, 226};allcolor["Blanc_céruse"] = {254, 254, 254};allcolor["Blanc_crème"] = {253, 241, 184};allcolor["Blanc_d_argent"] = {254, 254, 254};allcolor["Blanc_de_lait"] = {251, 252, 250};allcolor["Blanc_de_lin"] = {250, 240, 230};allcolor["Blanc_de_platine"] = {250, 240, 197};allcolor["Blanc_de_plomb"] = {254, 254, 254};allcolor["Blanc_de_Saturne"] = {254, 254, 254};allcolor["Blanc_de_Troyes"] = {254, 253, 240};allcolor["Blanc_de_Zinc"] = {246, 254, 254};allcolor["Blanc_d_Espagne"] = {254, 253, 240};allcolor["Blanc_d_ivoire"] = {255, 255, 212};allcolor["Blanc_écru"] = {254, 254, 224};allcolor["Blanc_lunaire"] = {244, 254, 254};allcolor["Blanc_neige"] = {254, 254, 254};allcolor["Blanc_opalin"] = {242, 255, 255};allcolor["Blanc-bleu"] = {254, 254, 254};allcolor["Coquille_d_oeuf"] = {253, 233, 224};allcolor["Cuisse_de_nymphe"] = {254, 231, 240};allcolor["Brun"] = {91, 60, 17};allcolor["Acajou"] = {136, 66, 29};allcolor["Alezan"] = {167, 103, 38};allcolor["Ambre"] = {240, 195, 0};allcolor["Auburn"] = {157, 62, 12};allcolor["Basané"] = {139, 108, 66};allcolor["Beige"] = {200, 173, 127};allcolor["Beige_clair"] = {245, 245, 220};allcolor["Beigeasse"] = {175, 167, 123};allcolor["Bistre"] = {61, 43, 31};allcolor["Bistre"] = {133, 109, 77};allcolor["Bitume"] = {78, 61, 40};allcolor["Blet"] = {91, 60, 17};allcolor["Brique"] = {132, 46, 27};allcolor["Bronze"] = {97, 78, 26};allcolor["Brou_de_noix"] = {63, 34, 4};allcolor["Bureau"] = {107, 87, 49};allcolor["Cacao"] = {97, 75, 58};allcolor["Cachou"] = {47, 27, 12};allcolor["Café"] = {70, 46, 1};allcolor["Café_au_lait"] = {120, 94, 47};allcolor["Cannelle"] = {126, 88, 53};allcolor["Caramel"] = {126, 51, 0};allcolor["Châtaigne"] = {128, 109, 90};allcolor["Châtain"] = {139, 108, 66};allcolor["Chaudron"] = {133, 83, 15};allcolor["Chocolat"] = {90, 58, 34};allcolor["Citrouille"] = {223, 109, 20};allcolor["Fauve"] = {173, 79, 9};allcolor["Feuille-morte"] = {153, 81, 43};allcolor["Grège"] = {187, 174, 152};allcolor["Gris_de_maure"] = {104, 94, 67};allcolor["Lavallière"] = {143, 89, 34};allcolor["Marron"] = {88, 41, 0};allcolor["Mordoré"] = {135, 89, 26};allcolor["Noisette"] = {149, 86, 40};allcolor["Orange_brûlée"] = {204, 85, 0};allcolor["Puce"] = {78, 22, 9};allcolor["Rouge_bismarck"] = {165, 38, 10};allcolor["Rouge_tomette"] = {174, 74, 52};allcolor["Rouille"] = {152, 87, 23};allcolor["Sang_de_boeuf"] = {115, 8, 0};allcolor["Senois"] = {141, 64, 36};allcolor["Sépia"] = {169, 140, 120};allcolor["Sépia"] = {174, 137, 100};allcolor["Tabac"] = {159, 85, 30};allcolor["Terre_de_Sienne"] = {142, 84, 52};allcolor["Terre_d_ombre"] = {98, 91, 72};allcolor["Terre_d_ombre"] = {146, 109, 39};allcolor["Vanille"] = {225, 206, 154};allcolor["Gris"] = {96, 96, 96};allcolor["Ardoise"] = {90, 94, 107};allcolor["Argent"] = {206, 206, 206};allcolor["Argile"] = {239, 239, 239};allcolor["Bis"] = {118, 111, 100};allcolor["Bistre"] = {61, 43, 31};allcolor["Bistre"] = {133, 109, 77};allcolor["Bitume"] = {78, 61, 40};allcolor["Céladon"] = {131, 166, 151};allcolor["Châtaigne"] = {128, 109, 90};allcolor["Etain_oxydé"] = {186, 186, 186};allcolor["Etain_pur"] = {237, 237, 237};allcolor["Fumée"] = {187, 210, 225};allcolor["Grège"] = {187, 174, 152};allcolor["Gris_acier"] = {175, 175, 175};allcolor["Gris_anthracite"] = {48, 48, 48};allcolor["Gris_de_Payne"] = {103, 113, 121};allcolor["Gris_fer"] = {132, 132, 132};allcolor["Gris_Fer"] = {127, 127, 127};allcolor["Gris_Perle"] = {206, 206, 206};allcolor["Gris_Perle"] = {199, 208, 204};allcolor["Gris_souris"] = {158, 158, 158};allcolor["Gris_tourterelle"] = {187, 172, 172};allcolor["Mastic"] = {179, 177, 145};allcolor["Pinchard"] = {204, 204, 204};allcolor["Plomb"] = {121, 128, 129};allcolor["Rose_Mountbatten"] = {153, 122, 144};allcolor["Taupe"] = {70, 63, 50};allcolor["Tourdille"] = {193, 191, 177};allcolor["Jaune"] = {255, 255, 0};allcolor["Ambre"] = {240, 195, 0};allcolor["Aurore"] = {255, 203, 96};allcolor["Beurre"] = {240, 227, 107};allcolor["Beurre_frais"] = {255, 244, 141};allcolor["Blé"] = {232, 214, 48};allcolor["Blond"] = {226, 188, 116};allcolor["Boutton_d_or"] = {252, 220, 18};allcolor["Bulle"] = {237, 211, 140};allcolor["Caca_d_oie"] = {205, 205, 13};allcolor["Chamois"] = {208, 192, 122};allcolor["Champagne"] = {251, 242, 183};allcolor["Chrome"] = {237, 255, 12};allcolor["Chrome"] = {255, 255, 5};allcolor["Citron"] = {247, 255, 60};allcolor["Fauve"] = {173, 79, 9};allcolor["Flave"] = {230, 230, 151};allcolor["Fleur_de_soufre"] = {255, 255, 107};allcolor["Gomme-gutte"] = {239, 155, 15};allcolor["Jaune_auréolin"] = {239, 210, 66};allcolor["Jaune_banane"] = {209, 182, 6};allcolor["Jaune_canari"] = {231, 240, 13};allcolor["Jaune_chartreuse"] = {223, 255, 0};allcolor["Jaune_de_cobalt"] = {253, 238, 0};allcolor["jaune_de_Naples"] = {255, 240, 188};allcolor["Jaune_d_or"] = {239, 216, 9};allcolor["Jaune_impérial"] = {255, 228, 54};allcolor["Jaune_mimosa"] = {254, 248, 108};allcolor["Jaune_moutarde"] = {199, 207, 0};allcolor["Jaune_nankin"] = {247, 226, 105};allcolor["Jaune_olive"] = {128, 128, 0};allcolor["Jaune_paille"] = {254, 227, 71};allcolor["Jaune_poussin"] = {247, 227, 95};allcolor["Maïs"] = {255, 222, 117};allcolor["Mars"] = {239, 209, 83};allcolor["Mastic"] = {179, 177, 145};allcolor["Miel"] = {218, 179, 10};allcolor["Ocre_jaune"] = {223, 175, 44};allcolor["Ocre_rouge"] = {221, 152, 92};allcolor["Or"] = {255, 215, 0};allcolor["Orpiment"] = {252, 210, 28};allcolor["Poil_de_chameau"] = {182, 120, 35};allcolor["Queue_de_vache"] = {195, 180, 112};allcolor["Queue_de_vache"] = {168, 152, 116};allcolor["Sable"] = {224, 205, 169};allcolor["Safran"] = {243, 214, 23};allcolor["Soufre"] = {255, 255, 107};allcolor["Topaze"] = {250, 234, 115};allcolor["Vanille"] = {225, 206, 154};allcolor["Vénitien"] = {231, 168, 84};allcolor["Noir"] = {0, 0, 0};allcolor["Aile_de_corbeau"] = {0, 0, 0};allcolor["Brou_de_noix"] = {63, 34, 4};allcolor["Cassis"] = {44, 3, 11};allcolor["Cassis"] = {58, 2, 13};allcolor["Dorian"] = {11, 22, 22};allcolor["Ebène"] = {0, 0, 0};allcolor["Noir_animal"] = {0, 0, 0};allcolor["Noir_charbon"] = {0, 0, 0};allcolor["Noir_d_aniline"] = {18, 13, 22};allcolor["Noir_de_carbone"] = {19, 14, 10};allcolor["Noir_de_fumée"] = {19, 14, 10};allcolor["Noir_de_jais"] = {0, 0, 0};allcolor["Noir_d_encre"] = {0, 0, 0};allcolor["Noir_d_ivoire"] = {0, 0, 0};allcolor["Noiraud"] = {47, 30, 14};allcolor["Réglisse"] = {45, 36, 30};allcolor["Orange"] = {237, 127, 16};allcolor["Abricot"] = {230, 126, 48};allcolor["Aurore"] = {255, 203, 96};allcolor["Bis"] = {241, 226, 190};allcolor["Bisque"] = {255, 228, 196};allcolor["Carotte"] = {244, 102, 27};allcolor["Citrouille"] = {223, 109, 20};allcolor["Corail"] = {231, 62, 1};allcolor["Cuivre"] = {179, 103, 0};allcolor["Gomme-gutte"] = {239, 155, 15};allcolor["Mandarine"] = {254, 163, 71};allcolor["Melon"] = {222, 152, 22};allcolor["Orangé"] = {250, 164, 1};allcolor["Orange_brûlée"] = {204, 85, 0};allcolor["Roux"] = {173, 79, 9};allcolor["Safran"] = {243, 214, 23};allcolor["Saumon"] = {248, 142, 85};allcolor["Tangerine"] = {255, 127, 0};allcolor["Tanné"] = {167, 85, 2};allcolor["Vanille"] = {225, 206, 154};allcolor["Ventre_de_biche"] = {233, 201, 177};allcolor["Rose"] = {253, 108, 158};allcolor["Bisque"] = {255, 228, 196};allcolor["Cerise"] = {222, 49, 99};allcolor["Chair"] = {254, 195, 172};allcolor["Coquille_d_oeuf"] = {253, 233, 224};allcolor["Cuisse_de_nymphe"] = {254, 231, 240};allcolor["Framboise"] = {199, 44, 72};allcolor["Fushia"] = {253, 63, 146};allcolor["Héliotrope"] = {223, 115, 255};allcolor["Incarnadin"] = {254, 150, 160};allcolor["Magenta"] = {255, 0, 255};allcolor["Magenta_foncé"] = {128, 0, 128};allcolor["Magenta_fushia"] = {219, 0, 115};allcolor["Mauve"] = {212, 115, 212};allcolor["Pêche"] = {253, 191, 183};allcolor["Rose_balais"] = {196, 105, 143};allcolor["Rose_bonbon"] = {249, 66, 158};allcolor["Rose_dragée"] = {254, 191, 210};allcolor["Rose_Mountbatten"] = {153, 122, 144};allcolor["Rose_thé"] = {255, 134, 106};allcolor["Rose_vif"] = {255, 0, 127};allcolor["Saumon"] = {248, 142, 85};allcolor["Rouge"] = {255, 0, 0};allcolor["Amarante"] = {145, 40, 59};allcolor["Bordeaux"] = {109, 7, 26};allcolor["Brique"] = {132, 46, 27};allcolor["Cerise"] = {187, 11, 11};allcolor["Corail"] = {231, 62, 1};allcolor["Ecarlate"] = {237, 0, 0};allcolor["Fraise"] = {191, 48, 48};allcolor["Fraise_écrasée"] = {164, 36, 36};allcolor["Framboise"] = {199, 44, 72};allcolor["Fushia"] = {253, 63, 146};allcolor["Grenadine"] = {233, 56, 63};allcolor["Grenat"] = {110, 11, 20};allcolor["Incarnadin"] = {254, 150, 160};allcolor["Incarnat"] = {255, 111, 125};allcolor["Magenta"] = {255, 0, 255};allcolor["Magenta_foncé"] = {128, 0, 128};allcolor["Magenta_fushia"] = {219, 0, 115};allcolor["Mauve"] = {212, 115, 212};allcolor["Nacarat"] = {252, 93, 93};allcolor["Ocre_rouge"] = {221, 152, 92};allcolor["Passe-velours"] = {145, 40, 59};allcolor["Pourpre"] = {158, 14, 64};allcolor["Prune"] = {129, 20, 83};allcolor["Rose_vif"] = {255, 0, 127};allcolor["Rouge_alizarine"] = {217, 1, 21};allcolor["Rouge_anglais"] = {247, 35, 12};allcolor["Rouge_bismarck"] = {165, 38, 10};allcolor["Rouge_bourgogne"] = {107, 13, 13};allcolor["Rouge_capucine"] = {255, 94, 77};allcolor["Rouge_cardinal"] = {184, 32, 16};allcolor["Rouge_carmin"] = {150, 0, 24};allcolor["Rouge_cinabre"] = {219, 23, 2};allcolor["Rouge_cinabre"] = {253, 70, 38};allcolor["Rouge_coquelicot"] = {198, 8, 0};allcolor["Rouge_cramoisi"] = {150, 0, 24};allcolor["Rouge_cramoisi"] = {220, 20, 60};allcolor["Rouge_d_Andrinople"] = {169, 17, 1};allcolor["Rouge_d_aniline"] = {235, 0, 0};allcolor["Rouge_de_falun"] = {128, 24, 24};allcolor["Rouge_de_mars"] = {247, 35, 12};allcolor["Rouge_écrevisse"] = {188, 32, 1};allcolor["Rouge_feu"] = {254, 27, 0};allcolor["Rouge_feu"] = {255, 73, 1};allcolor["Rouge_garance"] = {238, 16, 16};allcolor["Rouge_groseille"] = {207, 10, 29};allcolor["Rouge_ponceau"] = {198, 8, 0};allcolor["Rouge_rubis"] = {224, 17, 95};allcolor["Rouge_sang"] = {133, 6, 6};allcolor["Rouge_tomate"] = {222, 41, 22};allcolor["Rouge_tomette"] = {174, 74, 52};allcolor["Rouge_turc"] = {169, 17, 1};allcolor["Rouge_vermillon"] = {219, 23, 2};allcolor["Rouge_vermillon"] = {253, 70, 38};allcolor["Rouge-violet"] = {199, 21, 133};allcolor["Rouille"] = {152, 87, 23};allcolor["Sang_de_boeuf"] = {115, 8, 0};allcolor["Senois"] = {141, 64, 36};allcolor["Terracotta"] = {204, 78, 92};allcolor["Vermeil"] = {255, 9, 33};allcolor["Zizolin"] = {108, 2, 119};allcolor["Vert"] = {0, 255, 0};allcolor["Aigue-marine"] = {121, 248, 248};allcolor["Asperge"] = {123, 160, 91};allcolor["Bleu_sarcelle"] = {0, 142, 142};allcolor["Canard"] = {4, 139, 154};allcolor["Céladon"] = {131, 166, 151};allcolor["Givré"] = {128, 208, 208};allcolor["Glauque"] = {100, 155, 136};allcolor["Hooker"] = {27, 79, 8};allcolor["Jade"] = {135, 233, 144};allcolor["Kaki"] = {148, 129, 43};allcolor["Menthe"] = {22, 184, 78};allcolor["Menthe_à_l_eau"] = {84, 249, 141};allcolor["Sinople"] = {20, 148, 20};allcolor["Turquoise"] = {37, 253, 233};allcolor["Vert_absinthe"] = {127, 221, 76};allcolor["Vert_amande"] = {130, 196, 108};allcolor["Vert_anglais"] = {24, 57, 30};allcolor["Vert_anis"] = {159, 232, 85};allcolor["Vert_avocat"] = {86, 130, 3};allcolor["Vert_bouteille"] = {9, 106, 9};allcolor["Vert_chartreuse"] = {194, 247, 50};allcolor["Vert_citron"] = {0, 255, 0};allcolor["Vert_de_chrome"] = {24, 57, 30};allcolor["Vert_de_gris"] = {149, 165, 149};allcolor["Vert_de_vessie"] = {34, 120, 15};allcolor["Vert_d_eau"] = {176, 242, 182};allcolor["Vert_émeraude"] = {1, 215, 88};allcolor["Vert_empire"] = {0, 86, 27};allcolor["Vert_épinard"] = {47, 79, 79};allcolor["Vert_gazon"] = {58, 137, 35};allcolor["Vert_impérial"] = {0, 86, 27};allcolor["Vert_kaki"] = {121, 137, 51};allcolor["Vert_lichen"] = {133, 193, 126};allcolor["Vert_lime"] = {158, 253, 56};allcolor["Vert_malachite"] = {31, 160, 85};allcolor["Vert_mélèse"] = {56, 111, 72};allcolor["Vert_militaire"] = {89, 102, 67};allcolor["Vert_mousse"] = {103, 159, 90};allcolor["Vert_olive"] = {112, 141, 35};allcolor["Vert_opaline"] = {151, 223, 198};allcolor["Vert_perroquet"] = {58, 242, 75};allcolor["Vert_pin"] = {1, 121, 111};allcolor["Vert_pistache"] = {190, 245, 116};allcolor["Vert_poireau"] = {76, 166, 107};allcolor["Vert_pomme"] = {52, 201, 36};allcolor["Vert_prairie"] = {87, 213, 59};allcolor["Vert_prasin"] = {76, 166, 107};allcolor["Vert_printemps"] = {0, 255, 127};allcolor["Vert_sapin"] = {9, 82, 40};allcolor["Vert_sauge"] = {104, 157, 113};allcolor["Vert_smaragdin"] = {1, 215, 88};allcolor["Vert_tilleul"] = {165, 209, 82};allcolor["Vert_véronèse"] = {88, 111, 45};allcolor["Vert_viride"] = {64, 130, 109};allcolor["Violet"] = {102, 0, 153};allcolor["Améthyste"] = {136, 77, 167};allcolor["Aubergine"] = {55, 0, 40};allcolor["Bleu_persan"] = {102, 0, 255};allcolor["Byzantin"] = {189, 51, 164};allcolor["Byzantium"] = {112, 41, 99};allcolor["Cerise"] = {222, 49, 99};allcolor["Colombin"] = {106, 69, 93};allcolor["Fushia"] = {253, 63, 146};allcolor["Glycine"] = {201, 160, 220};allcolor["Gris_de_lin"] = {210, 202, 236};allcolor["Héliotrope"] = {223, 115, 255};allcolor["Indigo"] = {121, 28, 248};allcolor["Indigo"] = {46, 0, 108};allcolor["Indigo_du_web"] = {75, 0, 130};allcolor["Lavande"] = {150, 131, 236};allcolor["Lie_de_vin"] = {172, 30, 68};allcolor["Lilas"] = {182, 102, 210};allcolor["Magenta"] = {255, 0, 255};allcolor["Magenta_foncé"] = {128, 0, 128};allcolor["Magenta_fushia"] = {219, 0, 115};allcolor["Mauve"] = {212, 115, 212};allcolor["Orchidée"] = {218, 112, 214};allcolor["Parme"] = {207, 160, 233};allcolor["Pourpre"] = {158, 14, 64};allcolor["Prune"] = {129, 20, 83};allcolor["Rose_bonbon"] = {249, 66, 158};allcolor["Rose_vif"] = {255, 0, 127};allcolor["Rouge-violet"] = {199, 21, 133};allcolor["Violet_d_évêque"] = {114, 62, 100};allcolor["Violine"] = {161, 6, 132};allcolor["Zizolin"] = {108, 2, 119};
	/**finColor**/
	while(1){
		//we can loop the video by re-opening the capture every time the video reaches its last frame

		//capture.open("bouncingBall.avi");

		if(!capture.isOpened()){
			cout<<"ERROR ACQUIRING VIDEO FEED\n";
			getchar();
			return -1;
		}
		capture.set(CV_CAP_PROP_FRAME_WIDTH, 640);
   		capture.set(CV_CAP_PROP_FRAME_HEIGHT, 480);

		cameraWidth = 640;
		cameraHeight = 480;
		//check if the video has reach its last frame.
		//we add '-1' because we are reading two frames from the video at a time.
		//if this is not included, we get a memory error!
		 capture.read(frame2);
                 //convert frame2 to gray scale for frame differencing
                 cv::cvtColor(frame2,grayImage2,COLOR_BGR2GRAY);

		while(1){
			//read first frame
			capture.read(frame1);
			//convert frame1 to gray scale for frame differencing
			cvtColor(frame1,grayImage1,COLOR_BGR2GRAY);
			//perform frame differencing with the sequential images. This will output an "intensity image"
			//do not confuse this with a threshold image, we will need to perform thresholding afterwards.
			cv::absdiff(grayImage1,grayImage2,differenceImage);
			//threshold intensity image at a given sensitivity value
			cv::threshold(differenceImage,thresholdImage,SENSITIVITY_VALUE,255,THRESH_BINARY);
			if(debugMode==true){
				//show the difference image and threshold image
				cv::imshow("Difference Image",differenceImage);
				cv::imshow("Threshold Image", thresholdImage);
			}else{
				//if not in debug mode, destroy the windows so we don't see them anymore
				//cv::destroyWindow("Difference Image");
				//cv::destroyWindow("Threshold Image");
			}
			//blur the image to get rid of the noise. This will output an intensity image
			cv::blur(thresholdImage,thresholdImage,cv::Size(BLUR_SIZE,BLUR_SIZE));
			//threshold again to obtain binary image from blur output
			cv::threshold(thresholdImage,thresholdImage,SENSITIVITY_VALUE,255,THRESH_BINARY);
			if(debugMode==true){
				//show the threshold image after it's been "blurred"

				imshow("Final Threshold Image",thresholdImage);

			}
			else {
				//if not in debug mode, destroy the windows so we don't see them anymore
				//cv::destroyWindow("Final Threshold Image");
			}

			//if tracking enabled, search for contours in our thresholded image
			
			if(trackingEnabled){
				searchForMovement(thresholdImage,frame1);
			}

			//show our captured frame
			//imshow("Frame1",frame1);
			//check to see if a button has been pressed.
			//this 10ms delay is necessary for proper operation of this program
			//if removed, frames will not have enough time to referesh and a blank 
			//image will appear.
			switch(waitKey(10)){

			case 27: //'esc' key has been pressed, exit program.
				return 0;
			case 116: //'t' has been pressed. this will toggle tracking
				trackingEnabled = !trackingEnabled;
				if(trackingEnabled == false) cout<<"Tracking disabled."<<endl;
				else cout<<"Tracking enabled."<<endl;
				break;
			case 100: //'d' has been pressed. this will debug mode
				debugMode = !debugMode;
				if(debugMode == false) cout<<"Debug mode disabled."<<endl;
				else cout<<"Debug mode enabled."<<endl;
				break;
			case 112: //'p' has been pressed. this will pause/resume the code.
				pause = !pause;
				if(pause == true){ cout<<"Code paused, press 'p' again to resume"<<endl;
				while (pause == true){
					//stay in this loop until 
					switch (waitKey()){
						//a switch statement inside a switch statement? Mind blown.
					case 112: 
						//change pause back to false
						pause = false;
						cout<<"Code Resumed"<<endl;
						break;
					}
				}
				}



			}
		}
		//release the capture before re-opening and looping again.
		capture.release();
	}

	return 0;

}
