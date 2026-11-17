# 盤前研究：提早運算，06:00釋出

## 固定時序（Asia/Taipei）

- 前日19:17起分批取得公開行情、公司資訊、法人、營收與新聞，整點錯開17分啟動。
- 隔夜持續增量補歷史；每次最多120檔、FinMind最多400次請求，遇限流停止實際請求，不換帳戶或key。
- 03:27／04:27／05:17嘗試研究；每次依實際觀測時間凍結asOf，沿用同一策略與全市場排序。完成後僅上傳小型READY結果，較新的成功結果可替換尚未發布的舊準備結果。
- 05:40停止接收準備結果。06:00由Cloudflare Cron釋出；讀取API亦有相同時間閘門，可在Cron延遲時釋出已備妥結果。06:00前不能讀到當日未發布內容。
- 沒有有效READY時，只發布INCOMPLETE，不顯示昨日推薦為今日結果。發布不可改寫，不接受逾時補交或盤後回填。

GitHub schedule與Cloudflare均不是硬即時系統；這是將耗時工作移到發布前，非提供06:00秒級SLA。上游失敗、平台故障仍必須揭露。

## 儲存與權限

- GitHub標準Linux runner執行已bundle的研究程式，不需安裝App／Electron依賴。
- 本機SQLite checkpoint只存公開市場研究，不含使用者帳本、持股、TDCC或登入憑證。每次以Actions cache保存；每天02:17另備份壓縮checkpoint，保留2天作cache遺失復原。
- GitHub OIDC限定repository ID、owner ID、main分支、指定workflow、github-hosted runner、audience及短期token；沒有公開寫入或長期Cloudflare token。
- Cloudflare只保存prepared／publication、研究訊號與已到期紙上結果；不再執行全市場seed／feature／score工作。既有歷史表與不可變trigger保留，不刪資料。
- 真實已發布訊號由唯讀ledger接收，沿原executePaperTrade計算結果，分批20筆上傳；不憑新觀察補造過往推薦。

## 啟用順序

1. 跑相關測試、bundle與Worker dry-run。
2. 在研究D1套用0004_prepared_publication.sql；核實目標account／Worker／D1，不修改TDCC／情報服務。
3. 部署研究Worker與OIDC vars，RADAR_EXTERNAL_RESEARCH=true切停原重運算；公開gateway地址不變。
4. 僅提交研究bundle、workflow與研究migrations，不改既有情報workflow或其共用原始碼。
5. 真實Actions check驗證OIDC，再prepare檢查來源與cache，研究時窗內才可stage READY。
6. 下個平日06:00檢查session、asOf、preparedAt／publishedAt、processedCount、實際來源及watch／INCOMPLETE狀態。初次歷史回補仍需時間，不能把有seed紀錄數稱為全市場皆合格。

Workflow需要contents:read、actions:read、id-token:write；沒有部署或帳本權限。非成功結果保留上一份已驗證的當日prepared；不以舊日期結果補洞。
