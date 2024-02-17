#Cargo las Librerías
library(caret)
library(e1071)
library(e1071)
library(class)

#Cargo el dataset
datos<- as.data.frame(read.csv("C:/Users/maxco/Mi unidad/NoCountry/03-Codigo/Pruebas/card_transdata.csv", header=TRUE))
str(datos)

#paso la variable de clase a factor
datos$fraud <- as.factor(datos$fraud)
str(datos$fraud)

#Exploracion básica de la variable de clase
nFraude <- sum(datos$fraud==1) #87403 (0.09)
nNoFraude <- sum(datos$fraud==0) #912597 (91%)

#Porc de valores Fraude vs no Fraude
pFraude <- as.numeric(sprintf("%.2f", (sum(datos$fraud == 1) / nrow(datos))))
pNoFraude <- as.numeric(sprintf("%.2f", (sum(datos$fraud == 0) / nrow(datos))))

#Defino la lista con 1000 subdatasets 
vChuncks <- split(datos, rep(1:(nrow(datos) / 1000), each = 1000, length.out = nrow(datos)))

################### EJECUCION DE ALGORITMOS ###################

# 1 - SVM
#Recupero el modelo predicito SVN
#Siendo que el conjunto de datos es el dataset datos y SVM el modelo de aprendizaje
load('C:/Users/maxco/Mi unidad/NoCountry/03-Codigo/Backend/svmodel2.rda')

#Ejecución del modelo del primer chunck de a 1000 datos

P_svnmodel<-predict(svmmodel, vChuncks[[1]])
print(P_svnmodel)
summary(P_svnmodel)

#Presentación de la Informacion
confusionMatrix(P_svnmodel,vChuncks[[1]]$fraud)

# Obtener las etiquetas reales de los datos de prueba
etiquetas_reales <- vChuncks[[1]]$fraud

# Pongo Labels e Identifico VP, VN, FP, y FN
Vp_indices_SVM <- which(P_svnmodel == "1" & etiquetas_reales == "1")
Vn_indices_SVM <- which(P_svnmodel == "0" & etiquetas_reales == "0")
Fp_indices_SVM <- which(P_svnmodel == "1" & etiquetas_reales == "0")
Fn_indices_SVM <- which(P_svnmodel == "0" & etiquetas_reales == "1")

# Presento los resultado de la ejecución y eficacia del modelo
FP_SVM <- as.data.frame(vChuncks[[1]][Fp_indices_SVM, ])
FN_SVM <- as.data.frame(vChuncks[[1]][Fn_indices_SVM, ])
VP_SVM <- as.data.frame(vChuncks[[1]][Vp_indices_SVM, ])
VN_SVM <- as.data.frame(vChuncks[[1]][Vn_indices_SVM, ])

# 2 - GBOOST
#Recupero el modelo predictivo GBOOST
#Siendo que el conjunto de datos es el dataset datos y GBOOST el modelo de aprendizaje
load('C:/Users/maxco/Mi unidad/NoCountry/03-Codigo/Backend/boostFit2.rda')

#Ejecución del modelo del primer chunck de a 1000 datos

P_gBoost<-predict(boostFit, vChuncks[[1]])
print(P_gBoost)
summary(P_gBoost)

#Presentación de la Informacion
confusionMatrix(P_gBoost,vChuncks[[1]]$fraud)

# Obtener las etiquetas reales de los datos de prueba
etiquetas_reales <- vChuncks[[1]]$fraud

# Pongo Labels e Identifico VP, VN, FP, y FN
Vp_indices_GB <- which(P_gBoost == "1" & etiquetas_reales == "1")
Vn_indices_GB <- which(P_gBoost == "0" & etiquetas_reales == "0")
Fp_indices_GB <- which(P_gBoost == "1" & etiquetas_reales == "0")
Fn_indices_GB <- which(P_gBoost == "0" & etiquetas_reales == "1")

# Presento los resultado de la ejecución y eficacia del modelo
FP_GBoost <- as.data.frame(vChuncks[[1]][Fp_indices_GB, ])
FN_GBoost <- as.data.frame(vChuncks[[1]][Fn_indices_GB, ])
VP_GBoost <- as.data.frame(vChuncks[[1]][Vp_indices_GB, ])
VN_GBoost <- as.data.frame(vChuncks[[1]][Vn_indices_GB, ])

#3-Naive Bayes

#Siendo que el conjunto de datos es el dataset datos y NB el modelo de aprendizaje
load('C:/Users/maxco/Mi unidad/NoCountry/03-Codigo/Backend/NBmodel.rda')

#Ejecución del modelo del primer chunck de a 1000 datos
p_NB <- predict(m_naive, newdata = vChuncks[[1]]) 

print(p_NB)
summary(p_NB)

# Convertir los datos predichos a factor con los mismos niveles que los datos de referencia
p_NB_factor <- factor(p_NB, levels = levels(vChuncks[[1]]$fraud))

# Convertir los datos de referencia a factor
vChuncks_factor <- factor(vChuncks[[1]]$fraud, levels = levels(p_NB_factor))

# Calcular la matriz de confusión
confusionMatrix(p_NB_factor, vChuncks_factor)

# Obtener las etiquetas reales de los datos de prueba
etiquetas_reales <- vChuncks[[1]]$fraud

# Pongo Labels e Identifico VP, VN, FP, y FN
Vp_indices_NB <- which(p_NB == "1" & etiquetas_reales == "1")
Vn_indices_NB <- which(p_NB == "0" & etiquetas_reales == "0")
Fp_indices_NB <- which(p_NB == "1" & etiquetas_reales == "0")
Fn_indices_NB <- which(p_NB == "0" & etiquetas_reales == "1")

# Presento los resultado de la ejecución y eficacia del modelo
FP_NB <- as.data.frame(vChuncks[[1]][Fp_indices_NB, ])
FN_NB <- as.data.frame(vChuncks[[1]][Fn_indices_NB, ])
VP_NB <- as.data.frame(vChuncks[[1]][Vp_indices_NB, ])
VN_NB <- as.data.frame(vChuncks[[1]][Vn_indices_NB, ])
