# Rscript script/starGen.r -F fits/test -X 1024 -Y 1024 -N 2000 -A 0.5 -B 1 -C 700 -D 20000 -M cauchy -G 100 -S 30
# pouzivanie
# -F kam sa ulozi fits subor (zadava sa bez pripony fits), pdf-ka aj tablka s hviezdami sa ulozi do adresara odkial si skript spustil
# -X pocet stlpcov
# -Y pocet riadkov
# -N pocet hviezd
# -A min
# -B max hodnota sirky hviezdy (pre gauss je to stdev, pre cauchy je to hodnota gamma)
# -C min jasnost
# -D max jasnost
# -M metoda (gauss, cauchy)
# -G stredna hodnota sumu pozadia
# -S odchylka sumu pozadia


library(FITSio)
library(optparse)

option_list = list(
# used: ACDGHILMNOPWSXY
    make_option(c("-O", "--output"), 
                type    = "character", 
                default = NULL, 
                help    = "Output FITS file (path + name without extension .fits)"),
                
    make_option(c("-I", "--input"), 
                type    = "character", 
                default = NULL, 
                help    = "Input *.cat file with stars."),
                
    make_option(c("-H", "--header"), 
                type    = "character", 
                default = NULL, 
                help    = "FITS file that will be used for header."),
                
    make_option(c("-X", "--dimX"), 
                type    = "numeric", 
                default = 1024, 
                help    = "Number of columns (default 1024)"),
                
    make_option(c("-Y", "--dimY"), 
                type    = "numeric", 
                default = 1024, 
                help    = "Number of rows (default 1024)"),
                
    make_option(c("-P", "--prec"), 
                type    = "numeric", 
                default = 3, 
                help    = "Grid density of a pixel for star generation (default 3)"),
                
    make_option(c("-N", "--starCount"), 
                type    = "integer", 
                default = 1000, 
                help    = "Number of generated stars (default 1000) (30-500)"),
                
    make_option(c("-W", "--fwhm"), 
                type    = "numeric", 
                default = 3, 
                help    = "Full width at half maximum (default 5). 2-6(10)"),
                
#    make_option(c("-B", "--spreadMax"), 
#                type    = "numeric", 
#                default = NULL, 
#                help    = "max value of star width"),
#                
    make_option(c("-C", "--briMin"), 
                type    = "numeric", 
                default = 1000, 
                help    = "Min brightness (default 1000)"),
                
    make_option(c("-D", "--briMax"), 
                type    = "numeric", 
                default = 10000, 
                help    = "Max brightness (default 10000)"),
                
    make_option(c("-M", "--method"), 
                type    = "character", 
                default = "gauss", 
                help    = "Method: gauss, cauchy, line (default gauss)"),
                
    make_option(c("-L", "--length"), 
                type    = "numeric", 
                default = 3, 
                help    = "Specifies half-length of the object in gauss sigma (computed from fwhm), if method == line (default 3)"),
                
    make_option(c("-A", "--alpha"), 
                type    = "numeric", 
                default = 0, 
                help    = "Specifies slope of objects from 0 to 90, if method == line (default 0)"),
                
    make_option(c("-G", "--gaussM"), 
                type    = "numeric", 
                default = 100, 
                help    = "Mean value of background noise (default 100)"),
                
    make_option(c("-S", "--gaussS"), 
                type    = "numeric", 
                default = 30, 
                help    = "Stdev of gackground noise (default 30)")
)

opt_parser = OptionParser(option_list=option_list)
opt = parse_args(opt_parser)

# params
method    = opt$method
prec      = opt$prec
dimX      = opt$dimX*prec
dimY      = opt$dimY*prec
starCount = opt$starCount
briMin    = opt$briMin
briMax    = opt$briMax
backMean  = opt$gaussM
backSdev  = opt$gaussS

if(opt$method == 'line'){
    len   = opt$length
    alpha = opt$alpha*pi/180
}

if(!is.null(opt$header)){
    header = readFITS(opt$header)$header
}else{
    header = ''
}

parseCat = function(input){
    X = c()
    Y = c()
    F = c()
    W = c()
    alldata = readLines(input)
    for(i in 4:length(alldata)){
        l = gsub("^ *|(?<= ) | *$", "", alldata[i], perl = TRUE)
        p = strsplit(l, ' ')[[1]]
        if(length(p) != 17){
            next
        }
        X = c(X, as.numeric(p[12]))
        Y = c(Y, as.numeric(p[13]))
        F = c(F, as.numeric(p[14]))
        W = c(W, as.numeric(p[15]) / 1.67)
    }
    return(list(
        'X' = X,
        'Y' = Y,
        'F' = F,
        'W' = W))
}

switchX = function(F){
    return(apply(t(F[,ncol(F):1]),2,rev))
}

# bivariate gaussian
bigauss  = function(x,y,k,sigma2) k*exp(-0.5*(x**2 + y**2)/sigma2)

# bivariate cauchy
bicauchy = function(x,y,k,g2) k / (x**2 + y**2 + g2)**1.5

genStarGauss = function(star.x,star.y,bri,sigma){
    
    # x,y is a position of the star
    # bri is the birghtness of the whole object (ie, sum of its pixels)
    
    sigma2 = sigma**2
    k = 1/2/pi/sigma2
    lim = ceiling(5*sigma)
    
    # rows/y coordinates (upy <= y <= dwy) ... matrix orientation: TL = (1,1), BR = (dimX,dimY)
    upy =   floor(max(1,    star.y - lim))
    dwy = ceiling(min(dimY, star.y + lim))
    # cols/x coordinates (upx <= x <= dwx)
    upx =   floor(max(1,    star.x - lim))
    dwx = ceiling(min(dimX, star.x + lim))
    #cat(sigma)
    #cat("------")
    #cat(dwy)

    for(y in upy:dwy){ # rows
        for(x in upx:dwx){ # cols
            FITS[y,x] <<- FITS[y,x] + bri*bigauss(star.x - x + 0.5 , star.y - y + 0.5, k, sigma2)
        }
    }

    return(0)
}

genStarCauchy = function(star.x,star.y,bri,gam){
    
    # bri is the birghtness of the whole object
    
    k    = gam / 2 / pi
    gam2 = gam**2
    lim  = 5*gam
    
    # rows/y coordinates (upy <= y <= dwy) ... matrix orientation: TL = (1,1), BR = (dimX,dimY)
    upy =   floor(max(1,    star.y - lim))
    dwy = ceiling(min(dimY, star.y + lim))
    # cols/x coordinates (upx <= x <= dwx)
    upx =   floor(max(1,    star.x - lim))
    dwx = ceiling(min(dimX, star.x + lim))
    
    for(y in upy:dwy){ # rows
        for(x in upx:dwx){ # cols
            FITS[y,x] <<- FITS[y,x] + bri*bicauchy(star.x - x + 0.5, star.y - y + 0.5, k, gam2)
        }
    }
    
    return(0)
}

genLineGauss = function(star.x,star.y,bri,sigma,len,alpha){
    # x,y is a position of the star
    # bri is the birghtness of the whole object (ie, sum of its pixels)
    # len is the width of the plateua in units of sigma

    
            # check on which side of line a point (x,y) lies (in the direction of the line given by points A,B)
            
            checkRight = function(x,y,A,B){
                x1 = A[1]
                y1 = A[2]
                x2 = B[1]
                y2 = B[2]
                return((x - x1)*(y2 - y1) - (y - y1)*(x2 - x1) >= 0)
            }
#
            checkLeft  = function(x,y,A,B){
                x1 = A[1]
                y1 = A[2]
                x2 = B[1]
                y2 = B[2]
                return((x - x1)*(y2 - y1) - (y - y1)*(x2 - x1) <= 0)
            }
#
            inRange   = function(P,b){
                if(P < 1) P = 1
                if(P > b) P = b
                P
            }
            
            getCorners = function(C.X,C.Y,A,B,alpha){
                # corners of a box with centre C and half-length A and B
                # basic vectors of the box
                vecR = c(C.X, C.Y)
                vecA = c(A*cos(alpha), A*sin(alpha))
                vecB = c(B*cos(alpha+pi/2), B*sin(alpha+pi/2))
                
                # four corners of the box
                TL = vecR - vecA + vecB
                TR = vecR + vecA + vecB
                BL = vecR - vecA - vecB
                BR = vecR + vecA - vecB
                
                # define four bounding lines
                if(abs(tan(alpha)) > 0){
                    tgA  = tan(alpha)
                    tgA2 = tan(alpha + pi/2)
                    L1 = c(tgA,  -1, TR[2] - tgA *TR[1])
                    L2 = c(tgA,  -1, BL[2] - tgA *BL[1])
                    L3 = c(tgA2, -1, BL[2] - tgA2*BL[1])
                    L4 = c(tgA2, -1, TR[2] - tgA2*TR[1])
                }else{
                    L1 = c(0, 1, -TR[2])
                    L2 = c(0, 1, -BL[2])
                    L3 = c(1, 0, -BL[1])
                    L4 = c(1, 0, -TR[1])
                }
                
                # define box pixel borders (four lines perpendicular to axes and passing through box four corners)
                box.top = floor  (max(TL[2], TR[2], BL[2] ,BR[2]))   # first horizontal line from the top to cut the box
                box.bot = ceiling(min(TL[2], TR[2], BL[2] ,BR[2]))   # first horizontal line from the bottom to cut the box 
                box.lef = ceiling(min(TL[1], TR[1], BL[1] ,BR[1]))
                box.rig = floor  (max(TL[1], TR[1], BL[1] ,BR[1]))
                
                box.top = inRange(box.top, dimY)
                box.bot = inRange(box.bot, dimY)
                box.lef = inRange(box.lef, dimX)
                box.rig = inRange(box.rig, dimX)
            
                return(list('top' = box.top, 'bot' = box.bot, 'lef' = box.lef, 'rig' = box.rig))
            }
#
            projectPnt = function(P,A,B){
                # project point P onto line given by points A and B
                
                dot = function(a,b) sum(a*b)
                
                AP = P - A
                AB = B - A
                return(A + dot(AP,AB)/dot(AB,AB)*AB)
            }
#
    
    # centre of the object
    centre = c(star.x, star.y)
    
    # direction vector of the object (not normalized)
    shift  = len * sigma * c(cos(alpha), sin(alpha))
    
    # right and left point of the object (plateau boundaries)
    rightP = centre + shift
    leftP  = centre - shift
    
    # playground corners (it is a rotated box)
    corners = getCorners(star.x, star.y, A = len*sigma + 5*sigma, B = 5*sigma, alpha = alpha)
    
    # rows/y coordinates (upy <= y <= dwy) ... matrix orientation: TL = (1,1), BR = (dimX,dimY)
    upy = corners$bot
    dwy = corners$top
    
    # cols/x coordinates (upx <= x <= dwx)
    upx = corners$lef
    dwx = corners$rig
    
    NEWMAT = matrix(0, nrow=nrow(FITS), ncol=ncol(FITS))
    sigma2 = sigma**2

    alpha2 = alpha + pi/2
    SIN2 = sin(alpha2)
    COS2 = cos(alpha2)
    
    # two points on the centre line
    c1 = centre
    c2 = centre + c(COS2, SIN2)
    
    # two points on the left line
    l1 = leftP
    l2 = leftP + c(COS2, SIN2)
    
    # two points on the right line
    r1 = rightP
    r2 = rightP + c(COS2, SIN2)
    
    for(y in upy:dwy){ # rows
        for(x in upx:dwx){ # cols
            # left of left line
            if(checkLeft(x, y, l1, l2)){
                NEWMAT[y,x] = NEWMAT[y,x] + bigauss(leftP[1] - x + 0.5, leftP[2] - y + 0.5, 1, sigma2)
            }else{
            # right of right line
            if(checkRight(x, y, r1, r2)){
                NEWMAT[y,x] = NEWMAT[y,x] + bigauss(rightP[1] - x + 0.5, rightP[2] - y + 0.5, 1, sigma2)
            }else{
                # this is the strechted zone
                # project point on the centre line (given by points centre and centre + direction_vec)
                projected = projectPnt(c(x,y), c1, c2)
                NEWMAT[y,x] = NEWMAT[y,x] + bigauss(centre[1] - projected[1] + 0.5, centre[2] - projected[2] + 0.5, 1, sigma2)
            }}
        }
    }
    
    NEWMAT = bri * (NEWMAT / sum(NEWMAT))
    
    FITS <<- FITS + NEWMAT

    return(0)
}

if(is.null(opt$input)){
    # coordinates
    coordx = runif(starCount, 1, dimX)
    coordy = runif(starCount, 1, dimY)

    # brightness
    bri = runif(starCount, briMin, briMax)

    # star size
    if(method == 'gauss' | method == 'line'){
        sigma = rep(opt$fwhm / 2.355, starCount)
        sigma = sigma*prec
        sigma2 = sigma**2
    }
    if(method == 'cauchy'){
        gam = rep(opt$fwhm / 2, starCount)
        gam = gam * prec
    }
}else{
    
    input = parseCat(opt$input)
    
    cat('Parsed '); cat(as.character(length(input$X))); cat(' stars\n')
    
    starCount = length(input$X)
    
    # coordinates
    coordx = input$X*prec
    coordy = input$Y*prec

    # brightness
    bri = input$F
    
    if(is.null(opt$fwhm)){
        fwhm = input$W
    }else{
        fwhm = rep(opt$fwhm, length(input$W))
    }
    
    # star size
    if(method == 'gauss' | method == 'line'){
        sigma = fwhm / 2.355
        sigma = sigma*prec
    }
    if(method == 'cauchy'){
        gam = fwhm / 2
        gam = gam * prec
    }
}

# generate FITS
cat("Generating stars\n")
FITS = matrix(0, nrow=dimY, ncol=dimX)

# symmetrix gaussians
if(method == 'gauss'){
    for(i in 1:starCount){
        genStarGauss(coordx[i], coordy[i], bri[i], sigma[i])
    }
}

# symmetrix cauchians
if(method == 'cauchy'){
    for(i in 1:starCount){
        genStarCauchy(coordx[i], coordy[i], bri[i], gam[i])
    }
}

# lines
if(method == 'line'){
    for(i in 1:starCount){
        genLineGauss(coordx[i], coordy[i], bri[i], sigma[i], len, alpha)
    }
}

# write model - transform coordinates to original pixels
cat("Writing star table (model.tsv)\n")
model = cbind(coordx/prec, coordy/prec, bri)
colnames(model) = c('model.x', 'model.y', 'max.brightness')
write.table(model[order(model[,1]),], 'model.tsv', col.names=F, row.names=F, sep='\t')

# reduce to original size
if(prec > 1){
    cat("Reducing enahanced image\n")
    dimX = opt$dimX
    dimY = opt$dimY
    FITS_fin = matrix(0, nrow=dimY, ncol=dimX)
    for(y in 1:dimY){
        for(x in 1:dimX){
            y1 = (y-1)*prec + 1
            y2 = y*prec
            x1 = (x-1)*prec + 1
            x2 = x*prec
            FITS_fin[y,x] = sum(FITS[y1:y2,x1:x2])
        }
    }
    
    FITS = FITS_fin
}

# write model pdf
cat("Writing star image (model.pdf)\n")
pdf('model.pdf', height=10, width=10)
image(1:ncol(FITS), 1:nrow(FITS), t(FITS), col = grey(seq(0, 1, length = 65535)), axes = FALSE)
invisible(dev.off())

# generate poisson noise
cat("Generating Poisson noise\n")
for(x in 1:dimX){
    for(y in 1:dimY){
        FITS[x,y] = rpois(1, FITS[x,y])
    }
}

cat("Writing star FITS file ("); cat(paste(getwd(),'/',opt$output, '_stars.fits)\n', sep=''))
writeFITSim(switchX(FITS), file = paste(opt$output, '_stars.fits', sep=''), type = "single", 
            bscale = 1, bzero = 0, c1 = NA, c2 = NA, crpixn = NA, crvaln = NA, cdeltn = NA, ctypen = NA, cunitn = NA, axDat = NA, 
            header = header)
#

# generate gaussian background noise
cat("Generating Gaussian noise\n")
noise = matrix(abs(rnorm(dimX*dimY, mean=backMean, sd=backSdev)), nrow=dimY, ncol=dimX)
cat("Writing Gaussian noise (noise.pdf)\n")
pdf('noise.pdf', height=10, width=10)
image(1:ncol(noise), 1:nrow(noise), t(noise), col = grey(seq(0, 1, length = 65535)), axes = FALSE)
invisible(dev.off())

cat("Writing noise FITS file ("); cat(paste(getwd(),'/',opt$output, '_noise.fits)\n', sep=''))
writeFITSim(switchX(noise), file = paste(opt$output, '_noise.fits', sep=''), type = "single", 
            bscale = 1, bzero = 0, c1 = NA, c2 = NA, crpixn = NA, crvaln = NA, cdeltn = NA, ctypen = NA, cunitn = NA, axDat = NA, 
            header = header)
#

FITS = FITS + noise

makePosit = function(X){
    X[X < 0] = 0
    return(X)
}

for(i in 1:nrow(FITS)) FITS[i,] = makePosit(FITS[i,])

cat("Writing final image (stars.pdf)\n")
pdf('stars.pdf', height=10, width=10)
image(1:ncol(FITS), 1:nrow(FITS), t(FITS), col = grey(seq(0, 1, length = 65535)), axes = FALSE)
invisible(dev.off())

# write FITS
cat("Writing FITS file ("); cat(paste(getwd(),'/',opt$output, '.fits)\n', sep=''))
writeFITSim(switchX(FITS), file = paste(opt$output, '.fits', sep=''), type = "single", 
            bscale = 1, bzero = 0, c1 = NA, c2 = NA, crpixn = NA, crvaln = NA, cdeltn = NA, ctypen = NA, cunitn = NA, axDat = NA, 
            header = header)
#

cat('Done\n')
