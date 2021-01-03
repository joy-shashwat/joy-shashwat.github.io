$(function() {

    var $dashboard_mobile_toggle = $('#main_menu_mobile_toggle');
    var $document_width = $(document).width();
    var $mobile_width = 767;
    var $main_menu = $('#main_menu');
    var $main_menu_icons = $('#main_menu ul li i');
    var $main_menu_open_normal_class = 'desktop';
    var $main_menu_open_mobile_class = 'mobile_open';
    var $main_menu_closed_mobile_class = 'mobile';
    var $main_menu_second_layer = $('#main_menu ul li ul');
    var $main_content = $('#main_content');
    var $border_bottom_right_class = 'border-bottom-right';
    var $text_center_class = 'text-center';

    $dashboard_mobile_toggle.on('click', function(e){
        getDocumentSize();
        if($document_width <= $mobile_width){
            $main_menu.addClass($main_menu_closed_mobile_class);
            $main_menu.toggleClass($main_menu_open_mobile_class);
            $main_menu.toggleClass($border_bottom_right_class);
            $main_menu.addClass($text_center_class);
            if($main_menu.hasClass($main_menu_open_mobile_class)){
                $main_content.hide();
                $main_menu_second_layer.show();
                $main_menu_icons.hide();
            }else{
                $main_content.show();
                $main_menu_second_layer.hide();
                $main_menu_icons.show();
            }
        }
    });

    var resizeId;
    $(window).resize(function() {
        clearTimeout(resizeId);
        resizeId = setTimeout(setMenuSize, 200);
    });

    function getDocumentSize(){
        $document_width = $(document).width();
    }

    function setMenuSize(){
        getDocumentSize();
        $main_content.show();
        $main_menu_icons.show();
        $main_menu.removeClass($text_center_class+' '+$main_menu_open_normal_class+' '+$main_menu_open_mobile_class+' '+$main_menu_closed_mobile_class);
        if($document_width <= $mobile_width){
            $main_menu.addClass($main_menu_closed_mobile_class);
            $main_menu_second_layer.hide();
        }else{
            $main_menu.addClass($main_menu_open_normal_class);
            $main_menu_second_layer.show();
        }
    }

    setMenuSize();

});