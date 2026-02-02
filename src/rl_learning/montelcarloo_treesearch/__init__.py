# # browser.py
# import io
# from PIL import Image
# from playwright.async_api import async_playwright


# async def capture_screenshot_from_link(
#     link: str,
#     *,
#     width: int = 1280,
#     height: int = 800,
# ):
#     """
#     Opens a link in Chromium and returns:
#       - page object (for later actions)
#       - screenshot as PIL.Image (RGB)
#     """
#     playwright = await async_playwright().start()

#     browser = await playwright.chromium.launch(
#         headless=True,
#         args=["--no-sandbox", "--disable-dev-shm-usage"]
#     )

#     page = await browser.new_page(
#         viewport={"width": width, "height": height},
#         device_scale_factor=1
#     )

#     await page.goto(link, wait_until="load")

#     png_bytes = await page.screenshot(full_page=False)
#     image = Image.open(io.BytesIO(png_bytes)).convert("RGB")

#     return {
#         "playwright": playwright,
#         "browser": browser,
#         "page": page,
#         "image": image,
#     }


# # ui_tars_model.py
# import torch
# from transformers import AutoProcessor, AutoModelForImageTextToText
# from PIL import Image


# def load_ui_tars_model():
#     processor = AutoProcessor.from_pretrained(
#         "ByteDance-Seed/UI-TARS-1.5-7B",
#         trust_remote_code=True
#     )

#     model = AutoModelForImageTextToText.from_pretrained(
#         "ByteDance-Seed/UI-TARS-1.5-7B",
#         trust_remote_code=True,
#         torch_dtype=torch.float16,
#         device_map="auto"
#     )

#     return processor, model


# def run_ui_tars_on_screenshot(
#     screenshot: Image.Image,
#     text_prompt: str,
#     processor,
#     model,
#     max_new_tokens: int = 1024,
# ):
#     """
#     Returns the raw model answer (string).
#     """
#     messages = [
#         {
#             "role": "user",
#             "content": [
#                 {"type": "image", "image": screenshot},
#                 {"type": "text", "text": text_prompt},
#             ],
#         }
#     ]

#     inputs = processor.apply_chat_template(
#         messages,
#         add_generation_prompt=True,
#         tokenize=True,
#         return_tensors="pt",
#         return_dict=True,
#     ).to(model.device)

#     outputs = model.generate(
#         **inputs,
#         max_new_tokens=max_new_tokens,
#         do_sample=False,
#     )

#     answer = processor.decode(
#         outputs[0][inputs["input_ids"].shape[-1]:],
#         skip_special_tokens=True,
#     )

#     return answer

# # action_parsing.py
# from ui_tars.action_parser import (
#     parse_action_to_structure_output,
# )


# def parse_ui_tars_action(
#     answer: str,
#     screenshot,
#     *,
#     factor: int = 1000,
#     model_type: str = "qwen25vl",
# ):
#     """
#     Converts model answer into structured action dict.
#     """
#     img_w, img_h = screenshot.size

#     parsed = parse_action_to_structure_output(
#         answer,
#         factor=factor,
#         origin_resized_width=img_w,
#         origin_resized_height=img_h,
#         model_type=model_type,
#     )

#     return parsed

# # action_execution.py
# async def execute_actions_on_page(page, parsed_actions):
#     """
#     Executes parsed UI-TARS actions on a Playwright page.
#     Returns execution log.
#     """
#     results = []

#     for action in parsed_actions:
#         if action["action_type"] == "click":
#             x1, y1, x2, y2 = action["action_inputs"]["start_box"]

#             # UI-TARS gives boxes → convert to center point
#             x = int((x1 + x2) / 2)
#             y = int((y1 + y2) / 2)

#             await page.mouse.click(x, y)
#             results.append({"action": "click", "x": x, "y": y})

#         elif action["action_type"] == "type":
#             text = action["action_inputs"]["text"]
#             await page.keyboard.type(text)
#             results.append({"action": "type", "text": text})

#         # extend with scroll, keypress, etc.

#     return results


# async def agent_step(link: str, prompt: str):
#     # 1. Browser → screenshot
#     browser_ctx = await capture_screenshot_from_link(link)
#     page = browser_ctx["page"]
#     screenshot = browser_ctx["image"]

#     # 2. Model inference
#     processor, model = load_ui_tars_model()
#     answer = run_ui_tars_on_screenshot(
#         screenshot,
#         prompt,
#         processor,
#         model,
#     )

#     # 3. Parse actions
#     parsed = parse_ui_tars_action(answer, screenshot)

#     # 4. Execute actions
#     execution_log = await execute_actions_on_page(page, parsed)

#     return {
#         "answer": answer,
#         "parsed_actions": parsed,
#         "execution_log": execution_log,
#     }
