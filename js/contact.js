$(document).ready(function() {
	$('#contact-form').submit(function() {
		var hasError = false;
		
		$('#contact-form .error-message').remove();
		
		$('.requiredField').each(function() {
			if($.trim($(this).val()) == '') {
				var errorText = $(this).data('error-empty');
				$('#contact-form').append('<p class="error-message">'+errorText+'.</p>');
				$(this).addClass('inputError');
				hasError = true;
			} else if($(this).is("input[type='email']") || $(this).attr('name')==='email') {
				var emailReg = /^([\w-\.]+@([\w-]+\.)+[\w-]{2,4})?$/;
				if(!emailReg.test($.trim($(this).val()))) {
					var invalidEmail = $(this).data('error-invalid');
					$('#contact-form').append('<p class="error-message">'+invalidEmail+'.</p>');
					$(this).addClass('inputError');
					hasError = true;
				}
			}
		});
		
		if(!hasError) {
			var formInput = $(this).serialize();
			$.ajax({
			    url: $(this).attr('action'), 
			    method: "POST",
			    data: formInput,
			    dataType: "json",
			    success: function(data){
					$('#contact-form').append('<p class="error-message">You have successfully subscribed!</p>');
				}
			});
		}
		
		return false;	
	});
});