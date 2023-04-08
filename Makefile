# ifneq ($(KERNELRELEASE),) 
# obj-m := proclog.o
# else 
# KERNELDIR ?= /lib/modules/$(shell uname -r)/build 

# PWD := $(shell pwd)

# default: 
# 	$(MAKE) -C $(KERNELDIR) M=$(PWD) modules  
# endif 

# clean:
# 	$(MAKE) -C $(KERNELDIR) M=$(PWD) clean

obj-m += proclog.o

all:
	make -C /lib/modules/$(shell uname -r)/build M=$(PWD) modules

clean:
	make -C /lib/modules/$(shell uname -r)/build M=$(PWD) clean
	
test:
  	# We put a — in front of the rmmod command to tell make to ignore
  	# an error in case the module isn’t loaded.
	-sudo rmmod proclog
  	# Clear the kernel log without echo
	sudo dmesg -C
  	# Insert the module
	sudo insmod proclog.ko
  	# Display the kernel log
	dmesg