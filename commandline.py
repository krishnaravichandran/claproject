#import os
#os.system("ls")


from subprocess import Popen, PIPE
cmd1 = ['ls -l ~/']

p1 = Popen(cmd1 , shell=True, stdout=PIPE, stderr=PIPE)
out1, err1 = p1.communicate()
print("Return code: ", p1.returncode)
print(out1.rstrip(), err1.rstrip())


from subprocess import call
call('ls')
