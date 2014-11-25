#include "glm/glm.hpp"
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <string>

using namespace std;
using namespace glm;






#ifndef _SMALLOBJLOADER_H_
#define _SMALLOBJLOADER_H_

//#define _disp_info

class SmallObjMesh{
	string fileName;
	bool withTex;
public:
	vector<vector<float> > UVcoord;
	vector<vec3 > position;
	vector<vector<unsigned int> > faces;
	vector<vector<unsigned int> > UVfaces;

public:
	SmallObjMesh(string fn){
		withTex=false;
		if(fn.length()>0){
			fileName=fn;
			ifstream fin(fn.c_str());
			char line[80];
			
			while(fin.getline(line,80)){
				istringstream is(line);
				string first;
				is>>first;
				if(first.size()<1)continue;
				if(first[0]=='#') continue;
				else if(!strcmp(first.c_str(),"vt")) {
					withTex=true;
					vector<float> uv(2,0);
					is>>uv[0]>>uv[1];
					UVcoord.push_back(uv);
				}
				else if(!strcmp(first.c_str(),"f"))  {
					vector<unsigned int> fl(3,0);
					vector<unsigned int> uv(3,0);
					char buff[80];
					for(int i=0;i<3;i++){
						is.getline(buff,80,'/');
						istringstream face(buff);
						if(is.eof()){
							face>>fl[0]>>fl[1]>>fl[2];
							fl[0]--;
							fl[1]--;
							fl[2]--;
							faces.push_back(fl);
							break;
						}
						face>>fl[i];
						fl[i]--;
						//fout<<fl[i]<<' ';
						is.getline(buff,80,' ');
						face.clear();
						face.str(buff);
						//face.seekg();

						//face.read(buff,80);
						
						face>>uv[i];
						uv[i]--;
					}
					if(withTex){
						UVfaces.push_back(uv);
						faces.push_back(fl);
					}

				}
				else if(!strcmp(first.c_str(),"v")){
					vec3 coord;
					is>>coord[0]>>coord[1]>>coord[2];
					position.push_back(coord);

				}
			}
#ifdef _disp_info
			for(int i=0;i<position.size();i++){
				printf("[%d] v %f %f %f\n", i, position[i][0],position[i][1],position[i][2]);
			}	
			for(int i=0;i<UVcoord.size();i++){
				printf("[%d] vt %f %f\n", i,UVcoord[i][0],UVcoord[i][1]);
			}
			for(int i=0;i<faces.size()&&!withTex;i++){
				printf("[%d] f %d %d %d\n", i,faces[i][0]+1,faces[i][1]+1,faces[i][2]+1);
			}
			for(int i=0;i<faces.size()&&withTex;i++){
				printf("[%d] f %d/%d %d/%d %d/%d\n", i, faces[i][0]+1,UVfaces[i][0]+1,faces[i][1]+1,UVfaces[i][1]+1,faces[i][2]+1,UVfaces[i][2]+1);
			}
#endif
		}
		else{
			cerr<<"No file name specified!\n";
		}
	}




	void set(string fn){
		withTex=false;
		if(fn.length()>0){
			fileName=fn;
			ifstream fin(fn.c_str());
			char line[80];
			
			while(fin.getline(line,80)){
				istringstream is(line);
				string first;
				is>>first;
				if(first.size()<1)continue;
				if(first[0]=='#') continue;
				else if(!strcmp(first.c_str(),"vt")) {
					withTex=true;
					vector<float> uv(2,0);
					is>>uv[0]>>uv[1];
					UVcoord.push_back(uv);
				}
				else if(!strcmp(first.c_str(),"f"))  {
					vector<unsigned int> fl(3,0);
					vector<unsigned int> uv(3,0);
					char buff[80];
					for(int i=0;i<3;i++){
						is.getline(buff,80,'/');
						istringstream face(buff);
						if(is.eof()){
							face>>fl[0]>>fl[1]>>fl[2];
							fl[0]--;
							fl[1]--;
							fl[2]--;
							faces.push_back(fl);
							break;
						}
						face>>fl[i];
						fl[i]--;
						//fout<<fl[i]<<' ';
						is.getline(buff,80,' ');
						face.clear();
						face.str(buff);
						//face.seekg();

						//face.read(buff,80);
						
						face>>uv[i];
						uv[i]--;
					}
					if(withTex){
						UVfaces.push_back(uv);
						faces.push_back(fl);
					}

				}
				else if(!strcmp(first.c_str(),"v")){
					vec3 coord;
					is>>coord[0]>>coord[1]>>coord[2];
					position.push_back(coord);

				}
			}
#ifdef _disp_info
			for(int i=0;i<position.size();i++){
				printf("[%d] v %f %f %f\n", i, position[i][0],position[i][1],position[i][2]);
			}	
			for(int i=0;i<UVcoord.size();i++){
				printf("[%d] vt %f %f\n", i,UVcoord[i][0],UVcoord[i][1]);
			}
			for(int i=0;i<faces.size()&&!withTex;i++){
				printf("[%d] f %d %d %d\n", i,faces[i][0]+1,faces[i][1]+1,faces[i][2]+1);
			}
			for(int i=0;i<faces.size()&&withTex;i++){
				printf("[%d] f %d/%d %d/%d %d/%d\n", i, faces[i][0]+1,UVfaces[i][0]+1,faces[i][1]+1,UVfaces[i][1]+1,faces[i][2]+1,UVfaces[i][2]+1);
			}
#endif
		}
		else{
			cerr<<"No file name specified!\n";
		}
	}


};


#endif