import os

# s_dir=r"E:\BSR\BSDS500\data\images\test"
# o_dir=r"G:\Data\压缩包\Data\BSDS\Clean\images"
# iids = os.listdir(s_dir)
# #iids = iids[0:101]
# with open(o_dir + '/test.txt', 'w') as trnf:
#     for iid in iids:
#         iidfile=os.path.join(s_dir,iid)
#
#         trnf.write(iidfile + '\n')



def cddir(s_dir,o_dir):

    iids = os.listdir(s_dir)
    #iids = iids[0:101]
    with open(o_dir + '/test.txt', 'w') as trnf:
        for iid in iids:
            iidfile=os.path.join(s_dir,iid)

            trnf.write(iidfile + '\n')

path="/root/Data/DRIVE"
file_list = os.listdir(path)
for file in file_list:
    sub_file_list = os.listdir(os.path.join(path, file))
    for sub_file in sub_file_list:
        file_path = os.path.join(os.path.join(path, file), sub_file)
        if os.path.isdir(file_path):
            print("目录：", file_path)
            cddir(file_path, file_path)