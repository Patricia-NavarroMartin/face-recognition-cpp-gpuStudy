/*
 * BBDD_cpu_final.cpp
 *
 *  Created on: May 23, 2017
 *      Author: patri
 */

#include <dirent.h>
#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <stdio.h>
#include <sys/time.h>
#include <iomanip>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"

using namespace std;
using namespace cv;

//Variables globales
int label;
ofstream csv_file;
bool no_face_detected;
struct timeval empieza, acaba;

//Función para utilizar otros clasificadores predefinidos
vector<Rect> detect_default(Mat img, vector<Rect> faces)
{
	CascadeClassifier default_cascade;
	string default_path;
	string cascade_name;
	int default_num = 0;

	while(faces.size()!=1)
	{
		switch(default_num)
			{
			case 0:
				cascade_name="/haarcascades/haarcascade_frontalface_alt.xml";
				break;
			case 1:
					cascade_name="/haarcascades/haarcascade_frontalface_alt2.xml";
					break;
			case 2:
				cascade_name="/haarcascades/haarcascade_frontalface_default.xml";
					break;
			case 3:
				no_face_detected=true;
				return faces;
			}
			default_path="/home/ubuntu/Desktop" + cascade_name;
			default_cascade.load(default_path);
			default_cascade.detectMultiScale(img,faces,1.05,6);
			default_num ++;

	}
	cout<<"Rostro detectado con clasificador:"<<cascade_name<<endl;
	return faces;
}

//Función de detección y recorte de la cara en la imagen siendo analizada
void detect_crop(string imgpath, Mat img, string cascade_path)
{
	CascadeClassifier cascade;
	cascade.load(cascade_path);
    vector<Rect> faces;

    //Intentar detectar la cara con el clasificador especificado
    cascade.detectMultiScale(img,faces,1.4,6);

	if(faces.size()!=1)
	{
		//De no haber detectado con éxito el único rostro de la imagen, se ejecuta una función que prueba con los demás clasificadores predefinidos como último intento
    	faces=detect_default(img,faces);

		if(faces.size()!=1)
		{
			cout<<"Todavía no se encuentra 1 única cara en la imagen"<<endl;
            cout<<"ELIMINANDO:"<<imgpath<<endl;

            int deleted = remove(imgpath.c_str());
            if(deleted!=0)
            {
            	cerr<<"(!) ERROR ELIMINANDO"<<endl;
            }
            else
            {
            	cout<<"ELIMINACIÓN CORRECTA"<<endl;
            }
			no_face_detected=true;
			return;
		}

	}

    //Recortar y ajustar a tamaño estándar
    Rect face_area=faces[0];
    Mat image_cut = img(face_area);
    Size size(250,250);
    resize(image_cut, image_cut, size);
    vector<int> compression_params;
    compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
    compression_params.push_back(95);

    //Sobreescribimos la imagen original con la editada
    imwrite(imgpath,image_cut);

}

//Función que abre y modifica el fichero que contiene la BBDD
void explore(char *dir_name, string cascade_path)
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

			//Estudiar el tipo de elemento
			Mat grayimg;
			switch(entry->d_type)
			{
				case DT_DIR:
					//Si es otro directorio, se vuelve a explorar su contenido recursivamente
					explore((char*)subpath.c_str(), cascade_path);
					label++;
					break;

				case DT_REG:

					//Si es un archivo se lee la imagen directamente en formato de escala de grises (0)
					grayimg = imread(subpath,0);

					no_face_detected=false;
					detect_crop(subpath,grayimg,cascade_path);

					//Solo si ha encontrado una cara se procede a registrarla con su etiqueta
					if(no_face_detected==false)
					{
							csv_file << subpath<<";"<<label<<"\n";
							cout<<subpath <<";"<<label<<endl;
					}
					break;

				case DT_UNKNOWN:
					cout<<"Tipo de archivo desconocido:"<<subpath<<endl;
					break;
			}
		}
	}

	//Cerramos el directorio
	closedir(dir);
}

int main (int argc, char** argv)
{
	//Control de argumentos de entrada del programa
	if(argc!=4)
	{
		cerr<<"(!) Argumento no válido. Debe ser: create_csv <file_path> <csv_file_name.csv> <cascade_path> "<<endl;
		return -1;
	}
	else
	{
		gettimeofday(&empieza,NULL);

		//Definición de variables a partir de los argumentos de entrada
		string path = string(argv[1]);
		string csv_file_name = "/home/ubuntu/Desktop/BBDD-Results/"+string(argv[2]);
		string cascade_path = string(argv[3]);

		//Creación del log y notificación por consola
		csv_file.open(csv_file_name.c_str());
		cout<<"Dirección del directorio:"<<path<<endl;
		cout<<"Dirección del registro csv" <<csv_file_name<<endl;

		//Inicialización del etiquetado y comienzo de la edición
		label=0;
		explore((char*)path.c_str(),cascade_path);
		//Fin de edición
		csv_file.close();

	     //Cálculo de duración del programa total
	   	 gettimeofday(&acaba,NULL);
	   	 double us = (acaba.tv_usec-empieza.tv_usec);
	   	 double s = (acaba.tv_sec-empieza.tv_sec);
	   	 int min = (int)s/60;
	   	 int seg = (int)s%60;
	   	 double total = s + us/1000000;

	   	//Mostrar resultado por consola
	   	 cout << "Tiempo de ejecución total (segundos):  "<<total<<endl;
	   	 cout<< min <<" minutos "<<seg<<" segundos"<<endl;
	}
}


