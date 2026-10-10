import os

s_dir=r"E:\Python\Application\SRD-main\SOTS\nyuhaze500\hazy"
o_dir=r"E:\Python\Application\SRD-main\SOTS\nyuhaze500"
iids = os.listdir(s_dir)
#iids = iids[0:101]
with open(o_dir + '/test.txt', 'w') as trnf:
    for iid in iids:
        iidfile=os.path.join(s_dir,iid)

        trnf.write(iidfile + '\n')
# idds=[]
# with open(r"G:\Data\BSDS\Test.txt", 'r') as trnf:
#     for iid in trnf:
#         idds.append(iid.rstrip('\n'))
#
# s_dir=r"G:\Data\BSDS\Clean\Test"
# o_dir=r"G:\Data\BSDS\Clean"
# iids = os.listdir(s_dir)
# with open(o_dir + '/test.txt', 'w') as trnf:
#     for n in range(len(idds)):
#
#         iidfile=os.path.join(s_dir,idds[n])
#         trnf.write(iidfile + '\n')