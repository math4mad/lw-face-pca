"""
from CSV2 文件夹
返回训练和测试数据
"""


using CSV, DataFrames

csv_path = "/Users/lunarcheung/Downloads/lfw_funneled/CSV2"
arr = [i == 10 ? "lwface-pairs_10.txt" : "lwface-pairs_0$(i).txt" for i in 1:10]

df_arr = DataFrame[]  #df数组

data_prepare(str)=CSV.File("$csv_path/$str.csv") |> DataFrame

for str in arr
    local df = data_prepare(str)
    push!(df_arr, df)
end

#分表数据垂直合并成更多行的数据表
data = vcat(df_arr...)




"""
    export_face_data(n=25)
    返回 lw-face  训练和测试数据
    n为测试数据的张数
    return Xtr,ytr,Xte,yte
    Xtr (1600, 12000)
    Xte (1600, 25)
    Xtr,Xte 行为像素, 列为图片(样本或者观测数据)
TBW
"""
function  export_face_data(n=25)
   
    Xtr = data[:,2:end] |> Matrix |> transpose  #所有的数据作为训练数据
    ytr=data[:,1]
    Xte = data[1:n, 2:end] |> Matrix|>transpose     #取n张照片作为测试
    yte= data[1:n, 1]
     
    
    return Xtr,ytr,Xte,yte
    
end

export  export_face_data

