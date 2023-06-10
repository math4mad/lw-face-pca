""""
数据处理
合并图片数据和标签信息
返回的 csv 文件,第一列为标签信息
"""

import Images: load
import GLMakie: Axis
using GLMakie, MLJ, Images, ImageView, LinearAlgebra
using CSV, DataFrames

const current_dir=pwd()

directory_path = "/Users/lunarcheung/Downloads/lfw_funneled"
cd(directory_path)

arr = [i == 10 ? "pairs_10.txt" : "pairs_0$(i).txt" for i in 1:10]



function image_data(name)
    imgarr = [] #照片数组
    labelarr=[] #标签数组
    open(name, "r") do file
        for line in eachline(file)

            if isfile(line)
                tag=split(line,"/")[1]
                
                image_path = joinpath(directory_path, line)
                image = imresize(load(image_path), (40, 40))
                img_gray = Gray.(image)
                res = img_gray |> channelview |> Matrix .|> (Float64) |> d -> reshape(d, 1600, 1)
                push!(imgarr, res)
                push!(labelarr,tag)
                @info "success!"
            else
                @info("ERROR: Image not found!")
            end
        end
    end

    return imgarr,labelarr
end

function main()
    for name in arr
        imgarr,labelarr = image_data(name)
        Xtrain = reduce(hcat, imgarr)
        df1 = DataFrame(Xtrain', :auto)
        df2=DataFrame(label=labelarr)
        df=hcat(df2,df1) #水平合并照片 df 和标签df,第一列为标签

        #csv 文件不在当前项目文件夹中存在,位于下载图片文件夹中
        CSV.write("$(current_dir)/csv3/lwface-$(name).csv", df)
    end
end


main()
cd(current_dir)


