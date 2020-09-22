/*
 * face_recognition_final_version_TEST.cpp
 *
 *  Created on: May 23, 2017
 *      Author: patri

By downloading, copying, installing or using the software you agree to this license.
If you do not agree to this license, do not download, install,
copy or use the software.


                          License Agreement
               For Open Source Computer Vision Library
                       (3-clause BSD License)

Copyright (C) 2000-2015, Intel Corporation, all rights reserved.
Copyright (C) 2009-2011, Willow Garage Inc., all rights reserved.
Copyright (C) 2009-2015, NVIDIA Corporation, all rights reserved.
Copyright (C) 2010-2013, Advanced Micro Devices, Inc., all rights reserved.
Copyright (C) 2015, OpenCV Foundation, all rights reserved.
Copyright (C) 2015, Itseez Inc., all rights reserved.
Third party copyrights are property of their respective owners.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

  * Redistributions of source code must retain the above copyright notice,
    this list of conditions and the following disclaimer.

  * Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.

  * Neither the names of the copyright holders nor the names of the contributors
    may be used to endorse or promote products derived from this software
    without specific prior written permission.

This software is provided by the copyright holders and contributors "as is" and
any express or implied warranties, including, but not limited to, the implied
warranties of merchantability and fitness for a particular purpose are disclaimed.
In no event shall copyright holders or contributors be liable for any direct,
indirect, incidental, special, exemplary, or consequential damages
(including, but not limited to, procurement of substitute goods or services;
loss of use, data, or profits; or business interruption) however caused
and on any theory of liability, whether in contract, strict liability,
or tort (including negligence or otherwise) arising in any way out of
the use of this software, even if advised of the possibility of such damage.
 */

#include <dirent.h>
#include <string>
#include <iostream>
#include <sstream>
#include <fstream>
#include <iomanip>

#include "opencv2/core/core.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/gpu/gpu.hpp"

using namespace std;
using namespace cv;
using namespace cv::gpu;

//Variables globales
ofstream log_file;
int detect_num;
int default_num;
bool no_face_detected;
int im_width;
int im_height;
string carpeta;
string archivo;
Ptr<FaceRecognizer> model;
bool clahe;
int frame_num;
bool video;
bool cam;
int ID_num;
string str_s_label_ID;

Mat realce_clahe (Mat colorimg)
{
	Mat lab_img;
	cvtColor(colorimg, lab_img,CV_BGR2Lab);

	// Extraemos el canal de luz
	vector<cv::Mat> lab_planes(3);
	split(lab_img, lab_planes);  // tenemos L en lab_planes[0]

	Ptr<cv::CLAHE> clahe= cv::createCLAHE();
	clahe->setClipLimit(2);
	Mat img_clahe;
	clahe->apply(lab_planes[0],img_clahe);

	//Volvemos a meter el canal en la imagen tras ajustar su histograma
	img_clahe.copyTo(lab_planes[0]);
	merge(lab_planes, lab_img);

	//Volvemos a convertir en RGB
	Mat img_final;
	cvtColor(lab_img,img_final,CV_Lab2BGR);

	//Lo devuelve a color
	return img_final;
}

//Función que lee y clasifica el archivo csv en la imagen y la etiqueta correspondiente
void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator = ';') {
    std::ifstream file(filename.c_str(), ifstream::in);
    if (!file) {
        string error_message = "No se pudo abrir el archivo.";
        CV_Error(CV_StsBadArg, error_message);
    }
    string line, path, classlabel;
    while (getline(file, line)) {
        stringstream liness(line);
        getline(liness, path, separator);
        getline(liness, classlabel);
        if(!path.empty() && !classlabel.empty()) {
        	//Lectura en formato BGR
        	Mat photo=imread(path, 1);
        	if(clahe)photo = realce_clahe(photo);
        	cvtColor(photo, photo, CV_BGR2GRAY);
        	//Almacenamiento de imagenes en escala de grises y etiquetas en vectores
            images.push_back(photo);
            labels.push_back(atoi(classlabel.c_str()));
        }
    }
    file.close();
}

//Función para utilizar clasificadores predefinidos
GpuMat detect_default(GpuMat img, GpuMat faces)
{
	CascadeClassifier_GPU default_cascade;
	string default_path;
	string cascade_name;
	default_num = 0;

	while(detect_num!=1)
	{
		switch(default_num)
			{
			case 0:
				cascade_name="/haarcascades_cuda/haarcascade_frontalface_alt.xml";
				break;
			case 1:
					cascade_name="/haarcascades_cuda/haarcascade_frontalface_alt2.xml";
					break;
			case 2:
				cascade_name="/haarcascades_cuda/haarcascade_frontalface_default.xml";
					break;
			case 3:
				no_face_detected=true;
				return faces;
			}
			default_path="/home/ubuntu/Desktop" + cascade_name;
			default_cascade.load(default_path);
			detect_num=default_cascade.detectMultiScale(img,faces,1.05,6);
			default_num ++;
	}
	cout<<"Rostro detectado con clasificador:"<<cascade_name<<endl;
	return faces;
}

//Funcion que a partir de una imagen/frame detecta si hay 1 rostro y lo identifica con su etiqueta
Mat detection_and_recognition (Mat img, string imagen)
{
	//Ediciones de color previas al estudio
	if(clahe)img=realce_clahe(img);
	Mat grayimg;
	cvtColor(img, grayimg, CV_BGR2GRAY);

	//Reinicio de variables de alarma, traspaso de datos a GPU y detección por GPU
	no_face_detected=false;
	GpuMat faces;
	GpuMat gray_gpu(grayimg);
	detect_num=0;
	faces=detect_default(gray_gpu,faces);

	//Si no se ha reconocido ninguna cara no se procede a la fase de recortar
	if(no_face_detected)
	{
		cout<<"No se ha detectado ninguna cara en esta imagen:" <<imagen<<endl;
		if(video) log_file<<frame_num<<";"<<"NO FACE"<<"\n";
		else log_file<<carpeta<<";"<<archivo<<";"<<"NO FACE"<<"\n";
		return img;
	}

	//Descarga resultados de la GPU a la CPU
	Mat obj_host;
	faces.download(obj_host);
	Rect* cfaces = obj_host.ptr<Rect>();

	//Recortar y ajustar a tamaño estándar
	Rect face_area=cfaces[0];
	Mat face = grayimg(face_area);
	Mat face_resized;
	resize(face, face_resized, Size(im_width, im_height), 1.0, 1.0, INTER_CUBIC);

	//Se realiza la predicción, informando también de la distancia
	int predicted_label;
	double predicted_distance;
	model->predict(face_resized, predicted_label, predicted_distance);

	//Pasa la información numérica a strings
	ostringstream s_label;
	s_label<<predicted_label;

	ostringstream s_distance;
	s_distance<<predicted_distance;

	//Si se trata de un video/webcam, el resultado se graba también en formato video y sino solo se almacena en su log
	if(video)
	{
		Scalar color;
		//Según el resultado, cambia el color de identificación
		if(ID_num==predicted_label) color = CV_RGB(0,255,0);
		else if(predicted_label==-1) color = CV_RGB(255,255,0);
		else color = CV_RGB(255,0,0);


		//Crea el rectángulo que encuadrará la cara
		rectangle(img, face_area, color, 1);
		// Crea el texto de información
		string real_vs_predict = "Real = "+str_s_label_ID+" Prediccion = " + s_label.str();
		string distance_text = "Distancia ="+ s_distance.str();
		// Calcula la posición para no poner el texto dentro del rectángulo
		int pos_x = std::max(face_area.tl().x - 10, 0);
		int pos_y = std::max(face_area.tl().y - 10, 0);
		// Escribe en el frame
		putText(img, real_vs_predict, Point(pos_x, pos_y-25), FONT_HERSHEY_PLAIN, 1.0, color, 2.0);
		putText(img, distance_text, Point(pos_x, pos_y), FONT_HERSHEY_PLAIN, 1.0, color, 2.0);

		//Escribimos resultado en el log y por consola
		log_file << frame_num<<";"<<predicted_label<<";"<<predicted_distance<<"\n";
		cout<< frame_num<<";"<<predicted_label<<";"<<predicted_distance<<"\n";

		//Devuelve el frame editado
		return img;
	}
	else
	{
		//Escribe resultado en el log y por consola
		log_file << carpeta<<";"<<archivo<<";"<<predicted_label<<";"<<predicted_distance<<"\n";
		cout<< carpeta<<";"<<archivo<<";"<<predicted_label<<";"<<predicted_distance<<"\n";
		return img;
	}
}

//Función que abre y modifica el fichero a estudiar
void explore(char *dir_name)
{
	DIR *dir;
	struct dirent *entry;


	//Abre el directorio
	dir = opendir(dir_name);
	if(!dir)
	{
		cerr << "(!) No se encuentra el directorio" <<endl;
		return;
	}

	//Lectura del contenido mientras todavía haya algo que leer
	while((entry = readdir(dir)) != NULL)
	{
		if( entry->d_name[0] !='.')				// . indica que el nombre de la entrada es el mismo que el del directorio analizado
		{
			string subpath = string(dir_name) + "/" + string(entry->d_name);

			Mat img;
			//Estudiar el tipo de elemento
			switch(entry->d_type)
			{
				case DT_DIR:
					//Si es otro directorio, se vuelve a explorar su contenido recursivamente
					carpeta = string(entry->d_name);
					explore((char*)subpath.c_str());
					break;

				case DT_REG:
					archivo = string(entry->d_name);
					//Si es un archivo se lee la imagen a color
					img = imread(subpath,1);
					detection_and_recognition(img,subpath);
					break;
				case DT_UNKNOWN:
					cout<<"Tipo de archivo desconocido:"<<subpath<<endl;
					break;
			}
		}
	}
	//Cerrar el directorio
	closedir(dir);
}

//Funcion de conversión entre el valor int de la tabla ascii a su caracter real
char asciiNUM_a_char(int ascii)
{
	char c_num;
	if (ascii == 48) c_num='0';
	else if (ascii ==49) c_num = '1';
	else if (ascii ==50) c_num = '2';
	else if (ascii ==51) c_num = '3';
	else if (ascii ==52) c_num = '4';
	else if (ascii ==53) c_num = '5';
	else if (ascii ==54) c_num = '6';
	else if (ascii ==55) c_num = '7';
	else if (ascii ==56) c_num = '8';
	else if (ascii ==57) c_num = '9';

	return c_num;
}

//Funcion para cambiar la etiqueta a reconocer como correcta
bool nuevoID(int max_ID, bool valid_ID)
{

	//Nueva ventana para seleccionar la nueva identidad a reconocer
	Mat background;
	background= Mat::zeros(1000,1000,CV_8UC3);
	putText(background,"Nuevo ID (espacio = enter)",Point(50,150),FONT_HERSHEY_SIMPLEX,2.0,Scalar(255,255,255),2,8,false);
	imshow("Reconocimiento facial - ID",background);

	int c3 = waitKey();
	ostringstream s_label_ID;
	//Almacena número hasta presionar espacio
	while (c3!=' ')
	{
		//Si la tecla presionada es un número:
		if ((c3>47)&&(c3<58))
		{
			//Almacenamos el número. Convertimos de ascii a int.
			s_label_ID<< asciiNUM_a_char(c3);
			str_s_label_ID = s_label_ID.str();
			putText(background,str_s_label_ID,Point(100,250),FONT_HERSHEY_SIMPLEX,2.0,Scalar(255,255,255),2,8,false);

			ID_num=atoi(str_s_label_ID.c_str());

			imshow("Reconocimiento facial - ID",background);
		}
		else
		{
			background= Mat::zeros(1000,1000,CV_8UC3);
			putText(background,"Solo se admiten numeros",Point(50,450),FONT_HERSHEY_SIMPLEX,2.0,Scalar(0,0,255),2,8,false);
			imshow("Reconocimiento facial - ID",background);
			waitKey(3000);
			//Reinicio
			background= Mat::zeros(1000,1000,CV_8UC3);
			putText(background,"Nuevo ID (espacio = enter)",Point(50,150),FONT_HERSHEY_SIMPLEX,2.0,Scalar(255,255,255),2,8,false);
			imshow("Reconocimiento facial - ID",background);
		}
		c3=waitKey();
	}

	//Se analiza el ID introducido para ver si es válido o no.
	if (ID_num>max_ID)
	{
		cout<<"NO VALIDO "<<ID_num<<endl;
		ostringstream s_max_ID;
		s_max_ID<<max_ID;

		background= Mat::zeros(1000,1000,CV_8UC3);
		string errorID = "ID debe ser menor a " + s_max_ID.str();
		putText(background,errorID,Point(100,450),FONT_HERSHEY_SIMPLEX,2.0,Scalar(0,0,255),2,8,false);
		imshow("Reconocimiento facial - ID",background);
		waitKey(3000);

		valid_ID=false;
	}
	else
	{
		cout<<"VALIDO "<<ID_num<<endl;
		valid_ID=true;
	}

	destroyWindow("Reconocimiento facial - ID");
	return valid_ID;
}

//Función que abre el video de lectura y escribe el video resultante frame a frame.
void analisis_video(string nombre_resultados, string direccion_resultados, int max_ID)
{
	 Mat frame;

	 VideoCapture capture;
	 if(cam)capture.open(0);
	 else capture.open("/home/ubuntu/Desktop/video_demo.mp4");
	 if(!capture.isOpened())
	 {
	    			cerr << "(!) No se pudo abrir lector"<<endl;
	    			exit(1);
	 }

	//COMPROBACIONES PREVIAS AL VIDEOWRITER
	capture >> frame;
	if (frame.empty()) {
	    		cerr << "(!) Frame vacío\n";
	    		exit(1);
	}
	bool isColor = (frame.type() == CV_8UC3);
	int height = capture.get(CV_CAP_PROP_FRAME_HEIGHT);

	//DECLARACIÓN DEL VIDEOWRITER

	int codec = CV_FOURCC('M', 'J', 'P', 'G');
	double fps_write = 10.0;
	string filename = direccion_resultados +nombre_resultados+".avi";
	VideoWriter writer;
	writer.open(filename, codec, fps_write, frame.size(), isColor);
	//Control de errores en la creación
	if (!writer.isOpened()) {
	    		   cerr << "((!) No se pudo abrir el archivo para escribir.\n";
	    		   exit(1);
	 }

	 //____________INICIO DE LECTURA-ANÁLISIS-GRABACIÓN____________
	    for(;;)
	    {
	 	   //Captura del frame e iniciamos el cronómetro para el cálculo del fps
	       capture >> frame;

	  	 //Comprobar que no se ha terminado el video
	       if (!capture.read(frame)) {
	           cout << "Video de lectura finalizado\n";
	           break;
	       }

	       //Comienza la detección
	       ostringstream s_frame_num;
	       s_frame_num<<frame_num;
	       string imagen = "Frame_" + s_frame_num.str();
	       Mat frame_result = detection_and_recognition(frame,imagen);

			// Mostrar el resultado
			if(cam)
			{
				imshow("Reconocimiento facial", frame_result);
				int c = waitKey(1);
				if(c==27) return;
				else if (c==' ')
				{
					bool valid_ID=false;
					//Mientras no se introduzca un ID válido no sale del bucle.
					while(valid_ID==false)
					{
						//Cambiar el nº de ID a reconocer
						valid_ID=nuevoID(max_ID, valid_ID);
					}
				}

			}


			writer.write(frame_result);

	       frame_num++;
	    }
	    //____________FIN DE LECTURA-ANÁLISIS-GRABACIÓN____________
}


int main (int argc, char** argv)
{
	//Control de argumentos de entrada al programa
	setenv("DISPLAY", ":0",0);
	if(argc!=4)
	{
		cerr<< "(!) ERROR. Introduzca los siguientes argumentos: <csv_file_name.csv> <video_demo or webcam or file path> <results_log_name>"<<endl;
		return -1;
	}

    //Escoger si utilizar el ecualizador de histograma para tratar las imágenes.
    Mat background;
    string eq;
    background= Mat::zeros(1000,1000,CV_8UC3);
    putText(background,"Aplicar CLAHE? ",Point(100,150),FONT_HERSHEY_SIMPLEX,2.0,Scalar(255,255,255),2,8,false);
    putText(background,"0=No / 1=Si",Point(100,250),FONT_HERSHEY_PLAIN,2.0,Scalar(255,255,255),1,8,false);
    imshow("Reconocimiento facial - Configuraciones previas",background);
    int c0 = waitKey();
    switch(c0)
    {
		case 48:
			clahe = false;
			eq="SIN";
			break;
		case 49:
			clahe = true;
			eq="CON";
			break;
    }

	//A partir del archivo csv obtener las imagenes y las etiquetas
	string csv_file = string(argv[1]);
	cout<<"La BBDD empleada para el entrenamiento es:"<<csv_file<<endl;
    vector<Mat> images;
    vector<int> labels;
    try {
        read_csv(csv_file, images, labels);
    } catch (cv::Exception& e) {
        cerr << "(!) ERROR abriendo el archivo" << csv_file << "Motivo: " << e.msg << endl;
        exit(1);
    }

    //Almacenar tamaño estándar de las imágenes
    im_width = images[0].cols;
    im_height = images[0].rows;


    //Escoger el algoritmo del reconocedor facial
    string selected_recognizer;
    string umbral;
    background = Mat::zeros(1000,1000,CV_8UC3);
    putText(background,"Algoritmo: ",Point(100,150),FONT_HERSHEY_SIMPLEX,2.0,Scalar(255,255,255),2,8,false);
    putText(background,"1=Eigenfaces / 2=Fisherfaces / 3=LBPH",Point(100,250),FONT_HERSHEY_PLAIN,2.0,Scalar(255,255,255),1,8,false);
    imshow("Reconocimiento facial - Configuraciones previas",background);
    int c1 = waitKey();
    int c2;
    switch(c1)
    {
    case 49:
    	model = createEigenFaceRecognizer();
    	selected_recognizer = "Eigen";
    	break;
    case 50:
    	background = Mat::zeros(1000,1000,CV_8UC3);
    	putText(background,"Algoritmo: Fisherfaces",Point(100,150),FONT_HERSHEY_SIMPLEX,2.0,Scalar(255,255,255),2,8,false);
    	putText(background,"Aplicar umbral?",Point(100,250),FONT_HERSHEY_PLAIN,2.0,Scalar(255,255,255),1,8,false);
    	putText(background,"0=No / 1=Si",Point(100,350),FONT_HERSHEY_PLAIN,2.0,Scalar(255,255,255),1,8,false);
    	imshow("Reconocimiento facial - Configuraciones previas",background);
    	c2 = waitKey();
    	switch (c2)
    	{
			case 48:
					model = createFisherFaceRecognizer();
					umbral = "NO";
					break;
			case 49:
					if(clahe) model = createFisherFaceRecognizer(0,795.45);
					else model = createFisherFaceRecognizer(0,445.45);

					/*if(clahe) model = createFisherFaceRecognizer(0,754.45);
					else model = createFisherFaceRecognizer(0,427.27);*/

					umbral = "SI";
					break;
    	}
    	selected_recognizer = "Fisher";
    	break;
    case 51:
    	model = createLBPHFaceRecognizer();
    	selected_recognizer = "LBPH";
    	break;

    }

    model->train(images, labels);

    waitKey(3000);
    destroyWindow("Reconocimiento facial - Configuraciones previas");

    //Identificar si se realizará el reconocimiento a un video o conjunto de imágenes
    string modo;
    if(string(argv[2])=="video_demo")
    {
    	video = true;
    	cam= false;
    	modo="video";

    }
    else if(string(argv[2])=="webcam")
    {
        	video = true;
        	cam=true;
        	modo="webcam";

    }
    else
    {
    	video=false;
    	modo = "archivo";
    }

    cout<<modo<<endl;

    string direccion_resultados = "/home/ubuntu/Desktop/FR-CLAHE-COLOR/";
    string nombre_resultados=string(argv[3])+"_"+modo+"_"+selected_recognizer+"_"+eq+"_"+umbral;

    //Creación del log
    string log_file_name= direccion_resultados+nombre_resultados+".csv";
    cout<<log_file_name<<endl;
    log_file.open(log_file_name.c_str());
    cout<<"Dirección del registro (log) es:" <<log_file_name<<endl;
    if(log_file.is_open())cout << "Log abierto correctamente"<<endl;
    else
    {
    	cerr << "(!) ERROR. No se pudo acceder a esa dirección"<<endl;
    	exit(1);
    }

    //Comienza el estudio según el modo de ejecución
    if(video)
    {
        int num_labels=labels.size();
        int max_ID = labels[num_labels-1];

		bool valid_ID=false;
		while(valid_ID==false)
		{
			//Cambiar el nº de ID a reconocer
			valid_ID=nuevoID(max_ID, valid_ID);
		}


    	frame_num=0;
    	analisis_video(nombre_resultados, direccion_resultados,max_ID);
    }
    else
    {
    	string test_file =string(argv[2]);
    	explore((char*)test_file.c_str());
    }

	//Cerrar log
	log_file.close();
}



