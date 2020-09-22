/*
 * face_recognition_final_version_USUARIO.cpp
 *
 *  Created on: May 27, 2017
 *  Author: patri
 *
 *
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
 *
 *
 *
 *    */


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
int frame_num;
bool cam;
int ID_num;
string str_s_label_ID;

VideoCapture capture;
VideoWriter writer;
string modo;

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
        	//Lectura en formato escala de grises
        	Mat photo=imread(path, 0);
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
	detect_num=0;

	while(detect_num==0)
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
		cout<<"No se ha detectado una única cara en esta imagen:" <<imagen<<endl;
		if(modo =="directorio") log_file<<carpeta<<";"<<archivo<<";"<<"NO FACE"<<"\n";
		else log_file<<frame_num<<";"<<"NO FACE"<<"\n";

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

	Scalar color;
	string text;
	//Según el resultado, cambia el color de identificación
	if(ID_num==predicted_label)
	{
		color = CV_RGB(0,255,0);
		text = "CORRECTO";
	}
	else if(predicted_label==-1)
	{
		color = CV_RGB(255,255,0);
		text = "DESCONOCIDO";
	}
	else
	{
		color = CV_RGB(255,0,0);
		text= "OTRO";
	}

	//Crea el rectángulo que encuadrará la cara
	Rect face_area_text(face_area.tl().x +5, face_area.tl().y +5, face_area.width - 10, face_area.height - 10);
	rectangle(img, face_area_text, color, 1.5);
	// Crea el texto de información
	// Calcula la posición para no poner el texto dentro del rectángulo
	int pos_x = std::max(face_area.tl().x - 10, 0);
	int pos_y = std::max(face_area.tl().y - 10, 0);
	// Escribe en el frame
	putText(img, text, Point(pos_x, pos_y), FONT_HERSHEY_PLAIN, 1.0, color, 2.0);

	if(modo =="directorio")
	{
	    //Recortar y ajustar a tamaño estándar
	    img = img(face_area);
	    Size size(250,250);
	    resize(img, img, size);
		//Escribe resultado en el log y por consola
		log_file << carpeta<<";"<<archivo<<";"<<predicted_label<<";"<<predicted_distance<<"\n";
		cout<< carpeta<<";"<<archivo<<";"<<predicted_label<<";"<<predicted_distance<<"\n";
	}
	else
	{
		//Escribimos resultado en el log y por consola
		log_file << frame_num<<";"<<predicted_label<<";"<<predicted_distance<<"\n";
		cout<< frame_num<<";"<<predicted_label<<";"<<predicted_distance<<"\n";
	}

	return img;
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
			Mat resultado_DYR;
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
					resultado_DYR= detection_and_recognition(img,subpath);

					writer.write(resultado_DYR);
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
void analisis_video( int max_ID)
{
	 Mat frame;

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

//Función para acceder a los elementos de entrada al programa
bool prep_ENTRADAS(string modo, string direccion)
{
	bool prep_error = false;
	if(((modo == "webcam")||(modo =="video")))
	{
		if(modo == "webcam")capture.open(0);
		else capture.open(direccion);

		if(!capture.isOpened())
		{
			cerr << "(!) No se pudo abrir lector de video"<<endl;
			prep_error=true;
			return prep_error;

		}
		else cout<<"APERTURA CORRECTA"<<endl;
	}
	else
	{
		char *dir_name=(char*)direccion.c_str();
		DIR *dir;

		//Abre el directorio
		dir = opendir(dir_name);
		if(!dir)
		{
			cerr << "(!) No se encuentra el directorio" <<endl;
			prep_error=true;
			return prep_error;
		}
		else cout<<"APERTURA CORRECTA"<<endl;
	}
	return prep_error;
}

//Función para gestionar los elementos de salida del programa
bool prep_SALIDAS(string modo,string direccion_resultados,string nombre_resultados)
{
	bool prep_error=false;
	//Creación del log_________________________________________________________________
	string log_file_name= direccion_resultados+nombre_resultados+".csv";
	cout<<log_file_name<<endl;
	log_file.open(log_file_name.c_str());
	cout<<"Dirección del registro (log) es:" <<log_file_name<<endl;
	if(log_file.is_open())cout << "Log abierto correctamente"<<endl;
	else
	{
	    cerr << "(!) ERROR. No se pudo acceder a esa dirección del log"<<endl;
	    prep_error=true;
	    return prep_error;
	}

	//Video de salida_________________________________________________________________
	Mat frame;
	bool isColor;
	Size size;
	int codec = CV_FOURCC('M', 'J', 'P', 'G');
	double fps_write;
	string filename = direccion_resultados +nombre_resultados+".avi";


	if((modo == "webcam")||(modo =="video"))
	{
		capture >> frame;
		if (frame.empty()) {
			cerr << "(!) Frame vacío\n";
			prep_error=true;
			return prep_error;
		}
		else
		{
			isColor = (frame.type() == CV_8UC3);
			size = frame.size();
			fps_write = 10;
		}

	}
	else
	{
		isColor = true;
		size = Size(250,250);
		fps_write = 1;
	}
	writer.open(filename, codec, fps_write, size, isColor);
	//Control de errores en la creación
	if (!writer.isOpened()) {
		cerr << "((!) No se pudo abrir el archivo para escribir.\n";
		prep_error=true;
		return prep_error;
	}
	else cout<<"Video escritura configurado"<<endl;

	return prep_error;
}

int main (int argc, char** argv)
{
	//Control de argumentos de entrada al programa
	setenv("DISPLAY", ":0",0);

	string eq;
	Mat INTRO = imread("./UI/INTRO.png");
	Mat CARACT = imread("./UI/CARACTERISTICAS.png");
	Mat MODO_WEBCAM = imread("./UI/MODO_WEBCAM.png");
	Mat MODO_VIDEO = imread("./UI/MODO_VIDEO.png");
	Mat MODO_DIREC = imread("./UI/MODO_DIRECTORIO.png");
	Mat FIN = imread("./UI/FIN.png");

    //INICIO DEL PROGRAMA.
    cvNamedWindow("INICIO",CV_WINDOW_AUTOSIZE);
    imshow("INICIO",INTRO);
    waitKey(2000);
    imshow("INICIO",CARACT);
    waitKey(2000);

	//A partir del archivo csv obtener las imagenes y las etiquetas
    string csv_file = "./BBDD/db50_copy_gpu.csv";
    vector<Mat> images;
    vector<int> labels;
    try {
        read_csv(csv_file, images, labels);
    } catch (cv::Exception& e) {
        cerr << "(!) ERROR abriendo el archivo" << csv_file << "Motivo: " << e.msg << endl;
        return -1;
    }
    //Almacenar tamaño estándar de las imágenes
    im_width = images[0].cols;
    im_height = images[0].rows;


	//Estudiar argumento de entrada
    string direccion;
	if(argc == 1)
	{
    	modo ="webcam";
    	cam=true;
    	direccion = "-";
    	imshow("INICIO", MODO_WEBCAM);
	}
	else if (argc == 2)
	{
		cam = false;
		direccion = string(argv[1]);
		size_t puntos = direccion.find_last_of(".");		//Busca desde el final algun punto
		if(puntos == string::npos)
		{
			modo = "directorio";		//si no tiene extension es una carpeta
			imshow("INICIO", MODO_DIREC);
		}
		else
		{
			modo = "video";								//si tiene extension es un video
			imshow("INICIO", MODO_VIDEO);
		}
	}
	else
	{
		cerr<< "(!) ERROR. No introduzca argumento para el modo webcam, o introduzca la direccion del video/ directorio a emplear"<<endl;
		return -1;
	}

    waitKey(2000);
    string direccion_resultados = "./RESULTADOS/";
    string nombre_resultados=modo;

    //Preparaciones________________________________________________
    bool error_ENTRADAS =prep_ENTRADAS(modo,direccion);
    if (error_ENTRADAS) return -1;
    bool error_SALIDAS =prep_SALIDAS(modo,direccion_resultados,nombre_resultados);
    if (error_SALIDAS) return -1;

    //Entrenar algoritmo y al finalizar, destruir ventana__________
    model = createFisherFaceRecognizer(0,427.27);
    model->train(images, labels);
    destroyWindow("INICIO");

    //Comienza el estudio según el modo de ejecución
    int num_labels=labels.size();
    int max_ID = labels[num_labels-1];

	bool valid_ID=false;
	while(valid_ID==false)
	{
		//nº de ID a reconocer
		valid_ID=nuevoID(max_ID, valid_ID);
	}

	if(modo =="directorio")
	{
		explore((char*)direccion.c_str());
	}
	else
	{
		frame_num=0;
		analisis_video(max_ID);
	}

	//Cerrar log
	log_file.close();
	imshow("CIERRE", FIN);
	waitKey(2000);
}



