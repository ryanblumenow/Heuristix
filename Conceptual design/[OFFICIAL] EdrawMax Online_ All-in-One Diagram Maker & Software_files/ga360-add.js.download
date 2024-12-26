(function () {
    // <!-- Google Tag Manager Init -->
    // (function(w, d, s, l, i, u){
    //     w[l] = w[l] || [];
    //     w[l].push({
    //         'gtm.start': new Date().getTime(),
    //         event: 'gtm.js'
    //     });
    //     var f = d.getElementsByTagName(s)[0],
    //         j = d.createElement(s),
    //         dl = l != 'dataLayer' ? '&l=' + l : '',
    //         src = (document.location.search.match(/55gtmpreview=1/) || u=="" ? 'https://www.googletagmanager.com/gtm.js' : u) + '?id=' + i + dl
    //     j.async = true;
    //     j.src = src;
    //     f.parentNode.insertBefore(j, f);
    // })(window, document, 'script', 'dataLayer', gtmId,"");
    // <!-- End Google Tag Manager Init -->

    let getStandardTime = () => {
        let standTime = new Date().getTime() + (new Date().getTimezoneOffset() * 60 * 1000) + 8*60*60*1000,
            standDate = new Date(standTime),
            years = standDate.getFullYear(),
            mouth = (standDate.getMonth() + 1) < 10 ? '0' + (standDate.getMonth() + 1) : (standDate.getMonth() + 1),
            date = standDate.getDate() < 10 ? ('0' + standDate.getDate()) : standDate.getDate(),
            hour = standDate.getHours() < 10 ? ('0' + standDate.getHours()) : standDate.getHours(),
            minites = standDate.getMinutes() < 10 ? ('0' + standDate.getMinutes()) : standDate.getMinutes(),
            seconds = standDate.getSeconds() < 10 ? ('0' + standDate.getSeconds()) : standDate.getSeconds()
        return (years + '-' + mouth + '-' + date + '_' + hour + ':' + minites + ':' + seconds)
    }

    function getQueryVariable(href,variable){
        let vars = href.split("&");
        for (let i=0;i<vars.length;i++) {
            let pair = vars[i].split("=");
            if(pair[0] == variable){return pair[1];}
        }
        return(false);
    }
    function getCookie(name){
        let arr,reg = new RegExp("(^| )" + name + "=([^;]*)(;|$)");
        if (arr = document.cookie.match(reg)) {
            return unescape(arr[2]);
        } else {
            return null;
        }
    }
    function getParentTag(startTag,listArr={parentAttrList:[],parentClassList:[]}) {
        if (!(startTag instanceof HTMLElement)) return console.log('receive only HTMLElement');
        if ('BODY' !== startTag.parentElement.nodeName) {
            listArr.parentAttrList.push(startTag.parentElement.attributes)
            return getParentTag(startTag.parentElement,listArr)
        }else return listArr;
    }

    // var gtmId = "GTM-WPNBJKV"
    //////////////////////////

    window.dataLayer = window.dataLayer || [];
    var ga360InitData,
        initPageType =  document.querySelector("body").getAttribute("data-cat") || "",
        initPagePro =  document.querySelector("body").getAttribute("data-pro") || "",
        initPageLan =    document.querySelector("html").getAttribute("lang") || "",
        initPageOs =    document.querySelector("body").getAttribute("data-sys") || "",

        proId =    document.querySelector("body").getAttribute("pro-id") || "",
        proPrice =  document.querySelector("body").getAttribute("pro-rice") || "",
        proCategory = document.querySelector("body").getAttribute("pro-category") || "",
        proName = document.querySelector("body").getAttribute("pro-name") || "",
        proQuantity = document.querySelector("body").getAttribute("pro-quantity") || "",
        proVariant = document.querySelector("body").getAttribute("pro-variant") || ""
    
    window.dataLayer.push({//all page
        "domain": window.location.host,
        "product": initPagePro,
        "hitTimeStamp": getStandardTime(),
        "siteLanguange": initPageLan,
        "userAgent": navigator.userAgent,
        "userId": (getCookie("user_identity") && JSON.parse((unescape(getCookie("user_identity")))).id) || "" ///auth_uid
    })	

    if(initPageType === "product"){//产品页面
        ga360InitData = {
            "eeAction": "eeProductDetail",
            "hitTimeStamp": getStandardTime(),
            "productIds":[proId],
            "productNames":[proName],
            "productPrice":[proPrice],
            "productOs":[initPageOs],
            "proBrand":[initPagePro],
            "proQuantity":[proQuantity],
            "productVariant":[proVariant],
            "productCategory":[proCategory],
        }
    }else if(initPageType === "search-rusult"){//搜索结果页
        ga360InitData = {
            "cat55": "$siteSearchCategory",
            "hitTimeStamp": getStandardTime(),
            "kw55": getQueryVariable(window.location.href,'q')
        }
    }
    ga360InitData && window.dataLayer.push(ga360InitData)	

    // ******************************* wsc 异步加载 start
    var asynDom = document.querySelectorAll("body div.parameter")//link 跳转类型
    if(asynDom){
        for(let i = 0;i < asynDom.length;i++){
            if(asynDom[i].getAttribute("data-toggle") === "seasonal"){
                window.dataLayer.push({
                    event: "eventInt",
                    buttonName: "promotion",
                    location: "wsc_plugin_seasonal",
                    hitTimeStamp: getStandardTime(),
                    linkType:"campaign view"
                })
            }
            getAsynClass(asynDom[i].getAttribute("data-toggle"),i)//绑定点击
        }
    }

    function getAsynClass(c,i){
        var ga360timer = [],curAsyn = document.querySelectorAll("aside#wsc-plugin-" + c)
        ga360timer[i] = setTimeout(function () {
            if (curAsyn.length > 0) {
                if (typeof window.addEventListener === 'function'){ 
                    var list = curAsyn[0].querySelectorAll("[ga360name]")
                    for(let a = 0;a < list.length;a++){
                        if(c === "seasonal"){
                            list[a].setAttribute("ga360type","campaign click")      
                        }
                        (function (o) { 
                            list[a].addEventListener('click', function(e){ 
                                window.dataLayer.push({
                                    event: e.currentTarget.getAttribute("gaevent") || "buttonLink",
                                    buttonName: e.currentTarget.getAttribute("ga360name") || "",
                                    location: e.currentTarget.getAttribute("ga360location") || "",
                                    hitTimeStamp: getStandardTime(),
                                    targetUri:e.currentTarget.getAttribute("href") || "",
                                    linkType: e.currentTarget.getAttribute("ga360type") || "jump"   
                                })
                            }); 
                        })(list[a]);
                    }
                }
                clearTimeout(ga360timer[i])
            } else {
                getAsynClass(c)
            }
        }, 500)
    }
    // ******************************* wsc 异步加载 end

    var listLink = document.querySelectorAll('[ga360listname] a'),
        sectionLink = document.querySelectorAll('main>*'),
        eventArr = document.querySelectorAll('[gaEvent]'),eventListArr

    for(var a=0;a<listLink.length;a++){
        listLink[a].setAttribute("listGa",true)
    }
    for(var a=0;a<sectionLink.length;a++){
        for(var i=0;i<sectionLink[a].querySelectorAll("a").length;i++){//跳转类型
            if(sectionLink[a].querySelectorAll("a")[i].getAttribute("href") && !(sectionLink[a].querySelectorAll("a")[i].getAttribute("href").indexOf("#") > 0 || (sectionLink[a].querySelectorAll("a")[i].getAttribute("href").indexOf("javascript") >= 0))|| sectionLink[a].querySelectorAll("a")[i].getAttribute("data-href")){
                sectionLink[a].querySelectorAll("a")[i].setAttribute("ga360location","content_"+(a+1)+"_buttonLink_"+(i+1))

            }
        }
        for(var x=0;x<sectionLink[a].querySelectorAll("[gaEvent]").length;x++){//事件类型
            sectionLink[a].querySelectorAll("[gaEvent]")[x].setAttribute("ga360location","content_"+(a+1)+"_"+sectionLink[a].querySelectorAll("[gaEvent]")[x].getAttribute("gaEvent")+"_"+(x+1))
        }
    }

    var startTagl = document.querySelectorAll('a') 
    var attrListArr,proDownList = ['.exe', '.dmg', '.zip', '.pkg']
    for(var n=0;n<startTagl.length;n++){//link 跳转类型
        if((startTagl[n].getAttribute("href") && !(startTagl[n].getAttribute("href").indexOf("#") > 0 || (startTagl[n].getAttribute("href").indexOf("javascript") >= 0)))){
            attrListArr = getParentTag(startTagl[n]).parentAttrList
            if(startTagl[n].getAttribute("href").indexOf("download.wondershare.com")>-1){
                for(var d=0;d<proDownList.length;d++){

                    if(startTagl[n].getAttribute("href").indexOf(proDownList[d])>-1){
                        startTagl[n].setAttribute("ga360type","prd download")
                        startTagl[n].setAttribute("ga360proIds",startTagl[n].getAttribute("href").replace(/(.*\/)*([^.]+).*/ig,"$2").replace(/[^0-9]/ig,"").split("&"))
                        startTagl[n].setAttribute("ga360proNames",startTagl[n].getAttribute("href").replace(/(.*\/)*([^.]+).*/ig,"$2").replace(/[^(A-Za-z)]/ig,"").split("&"))
                        
                    }
                }
            }
            for (var i = 0; i <attrListArr.length; i++){ 
                for (var j = 0; j < attrListArr[i].length; j++){ 
                    var wsNav = attrListArr[i][j].nodeName === "class" && (attrListArr[i][j].nodeValue.indexOf("wsc-header2020-navbar-master")>=0) ,
                        proNav = attrListArr[i][j].nodeName === "class" && (attrListArr[i][j].nodeValue.indexOf("wsc-header2020-navbar-main")>=0) ,
                        topFooter = attrListArr[i][j].nodeName === "class" && (attrListArr[i][j].nodeValue.indexOf("wsc-footer2020-top")>=0) ,
                        bottomFooter = attrListArr[i][j].nodeName === "class" && (attrListArr[i][j].nodeValue.indexOf("wsc-footer2020-bottom")>=0) 

                    if(attrListArr[i][j].nodeName === "ga360listname"&&(startTagl[n].getAttribute("listGa"))){
                        startTagl[n].setAttribute("ga360Location", attrListArr[i][j].nodeValue + "_buttonLink_" + (n+1))
                    } else if(wsNav && (!startTagl[n].getAttribute("listGa"))){
                        startTagl[n].setAttribute("ga360Location", "nav_1_buttonLink_" + (n+1))
                    } else if(proNav && (!startTagl[n].getAttribute("listGa"))){
                        startTagl[n].setAttribute("ga360Location", "nav_2_buttonLink_" + (n+1))
                    }else if(topFooter && (!startTagl[n].getAttribute("listGa"))){
                        startTagl[n].setAttribute("ga360Location", "footer_1_buttonLink_" + (n+1))
                    } else if(bottomFooter && (!startTagl[n].getAttribute("listGa"))){
                        startTagl[n].setAttribute("ga360Location", "footer_2_buttonLink_" + (n+1))
                    }
                }
            }
        }
    }

    for(var a=0;a<eventArr.length;a++){//事件类型
        eventListArr = getParentTag(eventArr[a]).parentAttrList
        for (var b = 0; b <eventListArr.length; b++){ 
            for (var c = 0; c < eventListArr[b].length; c++){ 
                var wsNav = eventListArr[b][c].nodeName === "class" && (eventListArr[b][c].nodeValue.indexOf("wsc-header2020-navbar-master")>=0) ,
                    proNav = eventListArr[b][c].nodeName === "class" && (eventListArr[b][c].nodeValue.indexOf("wsc-header2020-navbar-main")>=0) ,
                    topFooter = eventListArr[b][c].nodeName === "class" && (eventListArr[b][c].nodeValue.indexOf("wsc-footer2020-top")>=0) ,
                    bottomFooter = eventListArr[b][c].nodeName === "class" && (eventListArr[b][c].nodeValue.indexOf("wsc-footer2020-bottom")>=0) 
                if(wsNav){
                    eventArr[a].setAttribute("ga360Location", "nav_1_"+eventArr[a].getAttribute("gaevent")+"_" + (a+1))
                } else if(proNav){
                    eventArr[a].setAttribute("ga360Location", "nav_2_"+eventArr[a].getAttribute("gaevent")+"_" + (a+1))
                }else if(topFooter){
                    eventArr[a].setAttribute("ga360Location", "footer_1_"+eventArr[a].getAttribute("gaevent")+"_" + (a+1))
                }else if(bottomFooter){
                    eventArr[a].setAttribute("ga360Location", "footer_2_"+eventArr[a].getAttribute("gaevent")+"_" + (a+1))
                }else{
                    eventArr[a].setAttribute("ga360Location", "other_plugin_"+eventArr[a].getAttribute("gaevent")+"_" + (a+1))
                }
            }
        }

    }

    // ******************************* 异步加载

    // addthis-smartlayers
    var addthisDom = document.querySelectorAll("body div.addthis-smartlayers #at4-share")//addthis 第三方
    if(addthisDom){
        for (var d = 0; d < addthisDom.length;d++){ 
            (function (o) { 
                for(var addthisDomA = 0;addthisDomA < addthisDom[d].querySelectorAll("a").length;addthisDomA++){
                    addthisDom[d].querySelectorAll("a")[addthisDomA].setAttribute("ga360location","addthis_plugin_"+ d + "_share_" + addthisDomA)
                    addthisDom[d].querySelectorAll("a")[addthisDomA].setAttribute("ga360type","share")
                    addthisDom[d].querySelectorAll("a")[addthisDomA].setAttribute("gaevent","eventInt")
                    addthisDom[d].querySelectorAll("a")[addthisDomA].setAttribute("ga360name",addthisDom[d].querySelectorAll("a")[addthisDomA].getAttribute("class"))
                }
                
            })(addthisDom[d]);
        }
    }
    // ******************************* 异步加载

    var linkType = document.querySelectorAll("a"),//link 跳转类型
    eventType = document.querySelectorAll("[gaEvent]")//事件类型
    if (typeof window.addEventListener === 'function'){ 
        if(linkType){//跳转类型
            for (var t = 0; t < linkType.length; t++){ 
                (function (o) { 
                    linkType[t].addEventListener('click', function(e){ 
                        // e.preventDefault()
                        if((e.currentTarget.href && !(e.currentTarget.href.indexOf("#") > 0 || (e.currentTarget.href.indexOf("javascript") >= 0))) || (e.currentTarget.getAttribute("data-href") && !(e.currentTarget.getAttribute("data-href").indexOf("#") > 0 || (e.currentTarget.getAttribute("data-href").indexOf("javascript") >= 0)))){
                            var aHref = e.currentTarget.href || e.currentTarget.getAttribute("data-href"),
                                ga360proNames = e.currentTarget.getAttribute("ga360proNames"),
                                ga360proIds = e.currentTarget.getAttribute("ga360proIds") || getQueryVariable(aHref,"pid"),
                                ga360proPrice = e.currentTarget.getAttribute("ga360proPrice"),
                                ga360proQty = e.currentTarget.getAttribute("ga360proQty") || getQueryVariable(aHref,"qty"),
                                ga360proBrand = e.currentTarget.getAttribute("ga360proBrand") || document.querySelector("body").getAttribute("data-pro"),
                                ga360proVariant = e.currentTarget.getAttribute("ga360proVariant"),
                                ga360proCategory = e.currentTarget.getAttribute("ga360proCategory"),
                                ga360userType = e.currentTarget.getAttribute("ga360userType"),
                                productOs = e.currentTarget.getAttribute("ga360proOs")
                            if(e.currentTarget.href.indexOf('sku_id') > -1){//链接含有sku_id,购买链接
                                var linkIntro = e.currentTarget.href.split('?')[1],linkJson = {};
                                linkIntro.forEach(function(item){
                                    var itemR = item.split('=');
                                    linkJson[itemR[0]] = itemR[1]
                                })
                                ga360proIds = linkJson['sku_id'];
                            }
                            window.dataLayer.push( {
                                "event": "buttonLink",
                                "hitTimeStamp": getStandardTime(),
                                "buttonName":e.currentTarget.getAttribute("ga360name")|| e.currentTarget.text || "",
                                "location": e.currentTarget.getAttribute("ga360location")||"",
                                "productName": (ga360proNames && ga360proNames.split(",")) || [],
                                "productId": ga360proIds && ga360proIds.split(",") || [],
                                "targetUri": e.currentTarget.href || e.currentTarget.getAttribute("data-href"),
                                "linkType":e.currentTarget.getAttribute("ga360type")||"jump",

                                "productIds": ga360proIds && ga360proIds.split(",") || [],
                                "productNames":(ga360proNames && ga360proNames.split(",")) || [],
                                "productPrice":(ga360proPrice && ga360proPrice.split(",")) || [],
                                "productOs":(productOs && productOs.split(",")) || [],
                                "proBrand":(ga360proBrand && ga360proBrand.split(",")) || [],
                                "proQuantity":(ga360proQty && ga360proQty.split(",")) || [],
                                "productVariant":(ga360proVariant && ga360proVariant.split(",")) || [],
                                "productCategory":(ga360proCategory && ga360proCategory.split(",")) || [],
                                "userType":(ga360userType && ga360userType.split(",")) || []
                            })
                        }
                    }); 
                })(linkType[t]);
            } 
        }
        if(eventType){//事件类型
            for (var i = 0; i < eventType.length; i++){ 
                (function (o) { 
                    eventType[i].addEventListener('click', function(e){ 
                        if(e.currentTarget.getAttribute("ga360download")){
                            window.dataLayer.push( {
                                "event": e.currentTarget.getAttribute("gaevent")||"",
                                "hitTimeStamp": getStandardTime(),
                                "buttonName": e.currentTarget.getAttribute("ga360name")|| e.currentTarget.text || e.currentTarget.innerHTML || "",
                                "location": e.currentTarget.getAttribute("ga360location")||"",
                                "linkType":e.currentTarget.getAttribute("ga360type")||"",
                                "targetUri": e.currentTarget.getAttribute("ga360download")
                            })
                        }else{
                            window.dataLayer.push( {
                                "event": e.currentTarget.getAttribute("gaevent")||"",
                                "hitTimeStamp": getStandardTime(),
                                "buttonName": e.currentTarget.getAttribute("ga360name")|| e.currentTarget.text || e.currentTarget.innerHTML || "",
                                "location": e.currentTarget.getAttribute("ga360location")||"",
                                "linkType":e.currentTarget.getAttribute("ga360type")||"",
                            })
                        }
                       
                    }); 
                })(eventType[i]);
            }
        }
    
    } 

    // 特殊处理 start

    // youtube 播放点击
    var videoYoutube = document.querySelectorAll("[data-toggle='youtube']")
    for(var i = 0;i < videoYoutube.length;i++){
        (function (o) { 
            videoYoutube[i].setAttribute("ga360location","video-youtube-" + (i+1))
            videoYoutube[i].addEventListener('click', function(e){ 
                window.dataLayer.push( {
                    "event": "click_video",
                    "hitTimeStamp": getStandardTime(),
                    "location": e.currentTarget.getAttribute("ga360location")||"",
                    "dataYoutube": e.currentTarget.getAttribute("data-youtube")||"",
                    "name":e.currentTarget.getAttribute("ga360name")||"",
                })
            }); 
        })(videoYoutube[i]);
    }

    // 特殊处理 end

})();