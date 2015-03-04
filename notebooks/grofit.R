require(grofit)

id = 'control'

t =  read.table(paste0("t_",id,".csv"), header = FALSE, sep = ",", dec = ".")
y =  read.table(paste0("y_",id,".csv"), header = FALSE, sep = ",", dec = ".")

plot(as.numeric(t[1,]),as.numeric(y[1,]), type='l')

y = cbind(rep(id,dim(y)[1]), rep(NA,dim(y)[1]), rep(0,dim(y)[1]), y)
#meany=as.data.frame(t(colMeans(y)))
#meany=cbind('assay',NA, 0, meany)

opt <- grofit.control(interactive=F)
model = grofit(time=t, data=y, ec50=F, opt)
write.csv(summary(model$gcFit), paste0("gcfit_",id,".csv"))
