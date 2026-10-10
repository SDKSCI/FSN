#include <vector>
#include <string>
#include<math.h>
#include<queue>
#include<iostream>
#include <numeric>
#include <algorithm>
#include <cfloat>

using namespace std;


struct RGB {

        unsigned char B;
        unsigned char G;
        unsigned char R;
};


vector<double> kseedsx;
vector<double> kseedsy;
vector<double> mseedsl;
vector<double> mseedsa;
vector<double> mseedsb;

vector<double> RL;
vector<double> GL;
vector<double> BL;

vector<int> allvec;
vector<int>  superpixellabel;


vector<int> EnforceLabelConnectivity(const int& numk,const int& SUPSZ,const int &height,const int &width)

{
	const int dx8[8] = { -1,-1, -1,  0, 0, 1,  1,1 };
	const int dy8[8] = { -1, 0,  1, -1, 1, 0, -1,1 };

	const int sz = width * height;
	queue<int>  xvec;
	queue<int>  yvec;
	allvec.clear();

	int relabel = 0;
	vector<int> nlabel(sz, -1);
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			if (nlabel[i * width + j] < 0)
			{

				nlabel[i * width + j] = relabel;
				while (!yvec.empty())
				{
					yvec.pop();
					xvec.pop();
				}
				yvec.push(i);
				xvec.push(j);

				int count = 1;
				while (!yvec.empty())
				{
					for (int n = 0; n < 8; n++)
					{

						int dx = xvec.front() + dx8[n];
						int dy = yvec.front() + dy8[n];


						if ((dx >= 0 && dx < width) && (dy >= 0 && dy < height))
						{
							if (nlabel[dy * width + dx] < 0 && superpixellabel[i * width + j] == superpixellabel[dy * width + dx])
							{
								//cout << "dy:" << dy << " dx:" << dx << endl;
								nlabel[dy * width + dx] = relabel;
								xvec.push(dx);
								yvec.push(dy);
								count++;
							}
						}
					}
					xvec.pop();
					yvec.pop();
				}
				allvec.push_back(count);
				relabel++;
			}

		}
	}
    mseedsl.assign(relabel, 0);
	mseedsa.assign(relabel, 0);
	mseedsb.assign(relabel, 0);
	kseedsx.assign(relabel, 0);
	kseedsy.assign(relabel, 0);

	vector<double> sigmal;
	vector<double> sigmaa;
	vector<double> sigmab;
	vector<double> sigmax;
	vector<double> sigmay;
	vector<double> inv;


	sigmal.assign(relabel, 0);
	sigmaa.assign(relabel, 0);
	sigmab.assign(relabel, 0);
	sigmax.assign(relabel, 0);
	sigmay.assign(relabel, 0);

	inv.assign(relabel, 0);
	allvec.assign(relabel, 0);

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			//cout<<i<<" "<<j << " "<<superpixellabel[i * width + j] <<endl;
			assert(nlabell[i * width + j] >=0);
			sigmal[nlabel[i*width+j]] += RL[i * width + j];
			sigmaa[nlabel[i*width+j]] += GL[i * width + j];
			sigmab[nlabel[i*width+j]] += BL[i * width + j];
			sigmax[nlabel[i*width+j]] += j;
			sigmay[nlabel[i*width+j]]+= i;

			allvec[nlabel[i*width +j]]++;
		}

	}

	for (int k = 1; k < relabel; k++)
	{
//        cout << "label " << k << " size: " << allvec[k] << endl;
		if (allvec[k] <= 0) { allvec[k] = 1; }
		inv[k] = 1.0 / double(allvec[k]);//computing inverse now to multiply, than divide later

	}

	for (int k = 0; k < relabel; k++)
	{

		mseedsl[k] = sigmal[k] * inv[k];
		mseedsa[k] = sigmaa[k] * inv[k];
		mseedsb[k] = sigmab[k] * inv[k];
		kseedsx[k] = sigmax[k] * inv[k];
		kseedsy[k] = sigmay[k] * inv[k];

	}








	vector<vector<int>>  adj(relabel, vector<int>());
	int xy;


	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{


			xy = nlabel[i * width + j];

			if (allvec[xy] < SUPSZ)
			{
				for (int n = 0; n < 8; n++)
				{
					int dy = i + dy8[n];
					int dx = j + dx8[n];
					if ((dx >= 0 && dx < width) && (dy >= 0 && dy < height))
					{
						if (nlabel[i * width + j] != nlabel[dy * width + dx])

						{

							if (allvec[nlabel[dy * width + dx]] >SUPSZ)
							{
								adj[xy].push_back(nlabel[dy * width + dx]);
							}


						}
					}
				}
			}
			sort(adj[xy].begin(), adj[xy].end(), greater<int>());
			adj[xy].erase(unique(adj[xy].begin(), adj[xy].end()), adj[xy].end());

		}
	}

	vector<int> mergenum(relabel, -1);
 	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{

			int acc = nlabel[i * width + j];

			if (allvec[acc] < SUPSZ)
			{
				if (mergenum[acc] == -1)
				{
					double mergemin = 1000;

					for (int k = 0; k < adj[acc].size(); k++)
					{


						double mergep =  sqrt((mseedsl[acc] - mseedsl[adj[acc][k]]) * (mseedsl[acc] - mseedsl[adj[acc][k]]) +
							(mseedsa[acc] - mseedsa[adj[acc][k]]) * (mseedsa[acc] - mseedsa[adj[acc][k]]) +
							(mseedsb[acc] - mseedsb[adj[acc][k]]) * (mseedsb[acc] - mseedsb[adj[acc][k]]));

						if (mergep < mergemin)
						{
							mergemin = mergep;
							mergenum[acc] = adj[acc][k];

						}


					}
					nlabel[i * width + j] = mergenum[acc];
				}
				else
				{
					nlabel[i * width + j] = mergenum[acc];
				}

			}
		}
	}

	return nlabel;
}

vector<int> getlabel(RGB *rgb_data, int* label_ptr, int w, int h,int n)
{


    int num=n;
    int width=w;
    int height=h;

    int sz= width*height;
    RL.assign(sz, 0);
    GL.assign(sz, 0);
    BL.assign(sz, 0);
    superpixellabel.assign(sz, 0);
    int max_label = -1;

    for (int i = 0; i < width * height; ++i) {

        RL[i] = rgb_data[i].R;
        GL[i] = rgb_data[i].G;
        BL[i] = rgb_data[i].B;
        superpixellabel[i] = label_ptr[i];
        if (label_ptr[i]+1 > max_label)
            {max_label = label_ptr[i]+1;}
    }


    vector<int>  xlabel ;
    double itern = 0.1;
    for (int j = 0; j<2; j++)
    { int size = (sz / num) * (itern);

        xlabel.clear();
        xlabel = EnforceLabelConnectivity(max_label,size,height,width);
         superpixellabel.clear();
         superpixellabel = xlabel;
    }


     return xlabel;


}
