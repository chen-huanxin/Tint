import os
import xlwt

LogFileDir = "/media/dell/564C2A944C2A6F45/Code/Supervised_contrastive_loss_pytorch-main/logs"

AllResultDir = "/media/dell/564C2A944C2A6F45/Code/Supervised_contrastive_loss_pytorch-main/result.xls"

book = xlwt.Workbook(encoding='utf-8',style_compression=0)
sheet = book.add_sheet('all_result',cell_overwrite_ok=True)
col = ('Name','Epoch','Accuracy','RMSE')
allcolum = 0

for i in range(0,4):
    sheet.write(allcolum,i,col[i])

all_log_files = os.listdir(LogFileDir)

for eachlog in all_log_files:

    Dirs_in_path = os.path.join(LogFileDir, eachlog)

    if os.path.isdir(Dirs_in_path):
        Dirs_in = os.listdir(Dirs_in_path)
    else:
        continue

    logfile = os.path.join(LogFileDir, eachlog, "log.txt")

    # If the file exists:
    if os.path.exists(logfile):
        # open the file
        filesize = os.path.getsize(logfile)
        print(logfile)
        print(filesize)
        if filesize > 5000:
        # Save The Valid result
            file = open(logfile, "r")

            all_lines = file.readlines()
            allcolum += 1
            Last_3_lines = all_lines[-4:]
            sheet.write(allcolum, 0, eachlog)
            for eachline in Last_3_lines:

                if "Best Accuracy" in eachline:
                    BestAccuracy = eachline.split(" ")

                    Accuracy = BestAccuracy[2]
                    sheet.write(allcolum, 2, Accuracy)

                    if len(BestAccuracy)>3:
                        rmse = BestAccuracy[4]
                        rmse = rmse[:-3]
                        sheet.write(allcolum, 3, rmse)

                if "Epoch" in eachline:
                    Epoch = eachline[5:7]
                    sheet.write(allcolum, 1, Epoch)

book.save(AllResultDir)



